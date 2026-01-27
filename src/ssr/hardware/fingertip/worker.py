import cv2
import numpy as np
from threading import Thread, Lock
from .tracker import Tracker
from . import marker_detection

def put_optical_flow_arrows_on_image(Backg, image, optical_flow, threshold=2.0):
    # 不影响原始图像
    image = image.copy()
    # 缩放光流
    scaled_flow = optical_flow * 1.0  # 缩放因子
    # 获取光流的起点和终点坐标
    flow_start = np.stack(
        np.meshgrid(range(0, scaled_flow.shape[1], 20), range(0, scaled_flow.shape[0], 20)), 2)
    flow_end = (scaled_flow[flow_start[:, :, 1], flow_start[:, :, 0], :] + flow_start).astype(np.int32)
    # 计算光流矢量的长度（范数）
    norm = np.linalg.norm(scaled_flow[flow_start[:, :, 1], flow_start[:, :, 0], :], axis=2)
    # 使用阈值过滤光流矢量
    norm[norm < threshold] = 0
    # 获取非零光流矢量的索引
    nz = np.nonzero(norm)
    # 将范数归一化到 0 到 255 之间
    diff = norm.max() - norm.min()
    if diff > 0:
        norm = np.asarray((norm - norm.min()) / diff * 255.0, dtype='uint8')
    else:
        norm = np.zeros_like(norm, dtype='uint8')
    # 绘制光流箭头
    for i in range(len(nz[0])):
        y, x = nz[0][i], nz[1][i]
        intensity = int(norm[y, x])
        # 红色深浅表示强度，(0, 0, intensity)
        cv2.arrowedLine(image,
                        pt1=tuple(flow_start[y, x]),
                        pt2=tuple(flow_end[y, x]),
                        color=(255 - intensity, 255 - intensity, 255),
                        thickness=2,
                        tipLength=.3)
    return image
def resize_crop_mini(img, imgw, imgh):
    # resize, crop and resize back
    img = cv2.resize(img, (320, 240))  # size suggested by janos to maintain aspect ratio
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
    img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    img = img[:, :-1]  # remove last column to get a popular image resolution
    img = cv2.resize(img, (imgw, imgh))  # final resize for 3d
    return img

def compute_tracker_gel_stats(thresh):
    numcircles = 9 * 7;
    mmpp = .063;
    true_radius_mm = .5;
    true_radius_pixels = true_radius_mm / mmpp;
    circles = np.where(thresh)[0].shape[0]
    circlearea = circles / numcircles;
    radius = np.sqrt(circlearea / np.pi);
    radius_in_mm = radius * mmpp;
    percent_coverage = circlearea / (np.pi * (true_radius_pixels) ** 2);
    return radius_in_mm, percent_coverage*100.

class CameraWorker(Thread):
    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.lock = Lock()
        self.data = None
        self.reset_flag = False

    def trigger_reset(self):
        with self.lock:
            self.reset_flag = True

    def stop(self):
        self.running = False
        self.join()

    def get_latest_data(self):
        with self.lock:
            return self.data

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

        tracker = Tracker(adaptive=True, cuda=False)
        ret, first_frame = cap.read()
        if not ret:
            print(f"Camera {self.camera_index}: Failed to read first frame")
            cap.release()
            return
            
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        while self.running:
            with self.lock:
                if self.reset_flag:
                    tracker.reset()
                    self.reset_flag = False
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 图像预处理
            processed_frame = resize_crop_mini(frame, 320, 240)
            
            # Mark点检测
            mask = marker_detection.find_marker(processed_frame)
            radius, coverage = compute_tracker_gel_stats(mask)
            
            # Mask显示图像
            mask_display = cv2.resize(mask.astype(np.uint8)*255, (640, 480))
            
            # 光流处理
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(first_gray, gray_frame)
            _, diff_threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            small_gray_frame = gray_frame
            flow = tracker.track(small_gray_frame)
            
            arrows = put_optical_flow_arrows_on_image(
                small_gray_frame, 
                cv2.cvtColor(small_gray_frame, cv2.COLOR_GRAY2BGR), 
                flow[15:-15, 15:-15, :]
            )
            
            with self.lock:
                self.data = {
                    'frame': frame,
                    'mask_display': mask_display,
                    'diff_threshold': diff_threshold,
                    'arrows': arrows,
                    'flow': flow,
                    'radius': radius,
                    'coverage': coverage
                }
            
            # 打印信息到终端
            # print(f"Cam {self.camera_index}: Radius={radius:.2f}mm, Coverage={coverage:.2f}%")
        
        cap.release()

if __name__ == "__main__":
    # 使用两个摄像头
    # 根据描述 /dev/video0, /dev/video1 是一组，/dev/video2, /dev/video3 是一组
    # 通常使用 0 和 2
    camera_indices = [0, 2]
    workers = []
    
    # 初始化
    for idx in camera_indices:
        # 创建窗口
        window_width = 320
        window_height = 240
        
        # cv2.namedWindow(f"frame_{idx}", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(f"frame_{idx}", window_width, window_height)

        cv2.namedWindow(f"arrows_{idx}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"arrows_{idx}", window_width, window_height)
        
        # cv2.namedWindow(f"mask_{idx}", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(f"mask_{idx}", 640, 480)
        
        # cv2.namedWindow(f"diff_{idx}", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(f"diff_{idx}", window_width, window_height)
        
        w = CameraWorker(idx)
        w.start()
        workers.append(w)
    
    print("Press 'q' to quit, 'r' to reset trackers.")

    try:
        while True:
            for w in workers:
                data = w.get_latest_data()
                if data:
                    idx = w.camera_index
                    # cv2.imshow(f'frame_{idx}', data['frame'])
                    # cv2.imshow(f'mask_{idx}', data['mask_display'])
                    # cv2.imshow(f'diff_{idx}', data['diff_threshold'])
                    cv2.imshow(f'arrows_{idx}', data['arrows'])
                    
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                for w in workers:
                    w.trigger_reset()
    finally:
        print("Stopping threads...")
        for w in workers:
            w.stop()
        cv2.destroyAllWindows()