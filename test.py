from ultralytics import YOLO
import cv2 as cv
import numpy as np
import dxcam

SCREEN_SIZE = (1920, 1080)
CONF_THRESHOLD = 0.1

model = YOLO('best.pt')



def main():
    # 初始化dxcam
    area = dxcam.create(output_idx=0, output_color="BGR")
    area.start(region=(0, 0, 1920, 1080), target_fps=30, video_mode=True)
    print("初始化dxcam成功")

    while True:
        # 获取帧
        frame = area.get_latest_frame()
        if frame is None:
            continue

        # 获取结果
        results = model(frame, stream=True, verbose=False, conf=CONF_THRESHOLD)
        for result in results:
            frame=result.plot()
            show_frame = cv.resize(frame, (960, 540)) 
            cv.imshow("YOLO Detection", show_frame)

        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()        
    area.stop()

if __name__ == '__main__':
    main()
