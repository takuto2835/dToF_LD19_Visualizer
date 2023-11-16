import cv2
import struct
import numpy as np
import serial
from serial.tools import list_ports
import threading
import time
from sklearn.cluster import DBSCAN
from pynput.mouse import Button, Controller

all_surroundings = []  # 一周のデータを保存するリスト
current_surrounding = []  # 現在の周囲データを保存するリスト
last_angle = 0  # 最後に確認した角度

is_running = True  # スレッドの実行状態を示す変数

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
center = (WINDOW_WIDTH // 2, 0)
initial_mask = None
mouse_state = 'up'
mouse = None

def create_mask_from_vertices(vertices, radius):
    """
    Create a binary mask image from the filtered vertices.
    A circle is drawn for each vertex with the given radius, and the area covered by the circles is considered foreground.
    """
    # Initialize an empty image
    mask_image = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)

    # Convert vertex positions to image coordinates and draw circles
    for vertex in vertices:
        x_int = int(vertex[0] + center[0])
        y_int = int(vertex[1] + center[1])
        cv2.circle(mask_image, (x_int, y_int), radius, 255, -1)  # -1 fills the circle

    # Convert the mask image to a boolean array
    mask = mask_image.astype(bool)

    return mask

def apply_mask_to_vertices(vertices, mask):
    """
    Applies a binary mask to the vertices. Only vertices within the mask's foreground are kept.
    """
    try:
        masked_vertices = []
        height, width = mask.shape  # maskの高さと幅を取得
        for vertex in vertices:
            x_int = int(vertex[0] + center[0])
            y_int = int(vertex[1] + center[1])
            # 範囲チェックを追加
            if 0 <= y_int < height and 0 <= x_int < width:
                if not mask[y_int, x_int]:  # Check if the vertex is within the mask's foreground
                    masked_vertices.append(vertex)
        return np.array(masked_vertices)
    except Exception as e:
        print(f"Error in apply_mask_to_vertices: {e}")
        return None

def overlay_mask_on_image(vis_image, mask, mask_color=(0, 255, 0), alpha=0.3):
    """
    Overlays a binary mask on top of the visualization image.

    :param vis_image: The original image on which to overlay the mask.
    :param mask: The binary mask to overlay.
    :param mask_color: The color to use for the mask overlay (default is green).
    :param alpha: The transparency factor for the mask overlay.
    """
    # Ensure vis_image is in color
    if len(vis_image.shape) == 2:  # Check if grayscale
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

    # Create a color image from the mask to overlay
    color_mask = np.zeros_like(vis_image, dtype=np.uint8)

    # We need to make sure that the mask is broadcastable to the color mask
    # This means we need to index where mask is True, and for these indices, set the color
    color_mask[mask] = mask_color
    # Blend the color mask with the image
    vis_image = cv2.addWeighted(color_mask, alpha, vis_image, 1 - alpha, 1)
    return vis_image


def get_lidar_frames_from_buffer(buffer):
    try:
        frames = []
        
        while len(buffer) >= 47:
            # ヘッダーが正しくない場合、次のヘッダーを探す
            if buffer[0] != 0x54:
                if 0x54 in buffer:
                    buffer = buffer[buffer.index(0x54):]
                else:
                    break
            
            # 一つのフレームを取り出す
            frame_data = buffer[:47]
            buffer = buffer[47:]
            
            # フレームのデータが正しいか確認
            if not check_lidar_frame_data(frame_data):
                continue
            
            frames.append(get_lidar_frame(frame_data))
        
        return frames, buffer
    except Exception as e:        
        print(f"Error occurred: {e}")



# LIDARフレームデータが正しいかを確認
def check_lidar_frame_data(data):
    return len(data) == 47 and data[1] == 0x2C 

# CRC8チェックサムの計算
def calc_crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x31
            else:
                crc <<= 1
            crc &= 0xFF
    return crc

def get_lidar_frame(data):
    try:
        frame = {}
        frame['header'] = data[0]
        frame['ver_len'] = data[1]
        frame['speed'] = struct.unpack("<H", bytes(data[2:4]))[0]
        frame['startAngle'] = struct.unpack("<H", bytes(data[4:6]))[0]
        
        points = []
        for i in range(12):
            start_index = 6 + 3 * i
            distance = struct.unpack("<H", bytes(data[start_index:start_index+2]))[0]
            intensity = data[start_index+2]
            points.append((distance, intensity))
        
        frame['points'] = points
        frame['endAngle'] = struct.unpack("<H", bytes(data[42:44]))[0]
        frame['timestamp'] = struct.unpack("<H", bytes(data[44:46]))[0]
        frame['crc8'] = data[46]
        
        return frame
    except Exception as e:        
        print(f"Error occurred: {e}")


def process_lidar_frames(frames):
    global current_surrounding, last_angle, all_surroundings

    for frame in frames:
        if frame['startAngle'] < last_angle:
            # 一周が完了したら、current_surroundingをall_surroundingsに追加
            all_surroundings.append(current_surrounding)
            current_surrounding = []
        
        current_surrounding.append(frame)
        last_angle = frame['startAngle']

def visualize_lidar_frames(frames):
    
    try:
        global initial_mask, mouse_state

        img = np.zeros((WINDOW_WIDTH, WINDOW_HEIGHT, 3), dtype=np.uint8)

        # 各点を格納するためのNumPy配列を初期化
        all_points = np.empty((0, 3), dtype=np.float32)  # x, y, intensityを格納

        for frame in frames:
            diff_angle = (frame['endAngle'] - frame['startAngle']) / 11.0
            if frame['endAngle'] <= frame['startAngle']:
                diff_angle += (36000.0 / 11.0)

            angles = ((frame['startAngle'] + np.arange(12) * diff_angle) % 36000) * (np.pi / 18000.0)
            distances = np.array([point[0] for point in frame['points']], dtype=np.float32)
            intensities = np.array([point[1] for point in frame['points']], dtype=np.float32)

            xs =  distances * 0.2 * np.cos(angles)
            ys = distances * 0.2 * np.sin(angles)
            points = np.vstack((xs, ys, intensities)).T

            all_points = np.vstack((all_points, points))

        # フィルタリング条件を設定
        x_min, x_max = -300, 300  # xの最小値と最大値
        y_min, y_max = 0, 300  # yの最小値と最大値

        # フィルタリングを適用
        filter_mask = (all_points[:, 0] >= x_min) & (all_points[:, 0] <= x_max) & \
                      (all_points[:, 1] >= y_min) & (all_points[:, 1] <= y_max)
        filtered_points = all_points[filter_mask]

        # 可視化
        for point in filtered_points:
            x, y, intensity = point
            cv2.circle(img, (int(center[0] + x), int(center[1] + y)), 2, (0, intensity, 255 - intensity), -1)

                # マスクが存在しない場合にのみ生成
        if initial_mask is None:
            initial_mask = create_mask_from_vertices(filtered_points, 20)
        
        img = overlay_mask_on_image(img, initial_mask)


        masked_points = apply_mask_to_vertices(filtered_points, initial_mask)

        if masked_points.size > 0:   
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=50, min_samples=5).fit(masked_points[:, :2])  # Consider only x and y for clustering
            labels = clustering.labels_

            # Find unique cluster labels and ignore noise if present
            unique_labels = set(labels)
            unique_labels.discard(-1)  # Remove noise label

            means = []

            # Compute and visualize centroids of clusters
            for k in unique_labels:
                class_member_mask = (labels == k)
                xy = masked_points[class_member_mask, :2]  # Extract cluster points
                centroid = xy.mean(axis=0)
                means.append(centroid)
                

            for mean in means:
                cv2.circle(img, (int(center[0] + mean[0]), int(center[1] + mean[1])), 5, (255, 255, 0), -1)

            if means:
                    mean = means[-1]
                    screen_x = int( mean[0] * 10.0 + 400) 
                    screen_y = int(mean[1] * 10.0) 
                    
                    mouse.position = (screen_x, screen_y)
                    print(mouse.position)
                    if mouse_state == 'up':
                        mouse_state = 'down'
                        time.sleep(0.1)  # Wait for the OS to catch up
                        mouse.click(Button.left, 1)
        else:
            mouse_state = 'up'

        cv2.imshow("Lidar Data", img)
    except Exception as e:        
        print(f"Error occurred: {e}")

def read_data_from_serial():
    # 利用可能なシリアルポートをリストアップ
    available_ports = list(serial.tools.list_ports.comports())

    if not available_ports:
        print("利用可能なシリアルポートが見つかりません。プログラムを終了します。")
    else:
        # 一番最初のシリアルポートを選択
        selected_port = available_ports[0].device
        print(f"選択したシリアルポート: {selected_port}")

        # シリアルポートに接続
        try:
            ser = serial.Serial(selected_port, 230400, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
            time.sleep(1)
            buffer = bytearray()
            global is_running
            while is_running:  # この条件を追加
                while ser.in_waiting:
                    buffer.extend(ser.read(ser.in_waiting))
                frames, buffer = get_lidar_frames_from_buffer(buffer)
                if frames:
                    process_lidar_frames(frames)
                    #all_frames.extend(frames)
        except serial.SerialException as e:
            print(f"シリアルポートの接続中にエラーが発生しました: {e}")
    
    
def visualize_lidar_data_continuously():
    try:
        global is_running
        while is_running:  # この条件を追加
            if all_surroundings and len(all_surroundings[-1]) > 5:
                visualize_lidar_frames(all_surroundings[-1])
                all_surroundings.clear()
            time.sleep(0.01)

            key = cv2.waitKey(1)
            
            if key == ord('q') or key == 27:  # qキーまたはESCキーが押されたら
                print("end!")
                is_running = False
    except Exception as e:        
        print(f"Error occurred: {e}")

def main():
    global is_running , mouse

    try:
        mouse = Controller()
    except Exception as e:
        print(f"Failed to initialize mouse controller: {e}")
        exit(1)

    try:

        # データの読み取りを行うスレッドを開始
        data_thread = threading.Thread(target=read_data_from_serial)
        data_thread.start()

        # 可視化を行うスレッドを開始
        visualization_thread = threading.Thread(target=visualize_lidar_data_continuously)
        visualization_thread.start()

    except Exception as e:        
        print(f"Error occurred: {e}")
    finally:
        data_thread.join()
        visualization_thread.join()
    
        cv2.destroyAllWindows()  # OpenCVのウィンドウをすべて閉じる
        print("All thread is ended.")

if __name__ == "__main__":
    main()