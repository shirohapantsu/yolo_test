"""
yolo_in_k230 的 Docstring
-------------------------------

在k230上部署yolov8实现detect医疗包，画面输出至ide

author:shiroha_pantsu
date:2026-01-21
"""

import os, time, sys, image, random, gc
from media.display import *
from media.media import *
from media.sensor import *
from Libs.YOLO import YOLOv8
import nncase_runtime as nn
import ulab.numpy as np

# 配置模型参数
kmodel_path = "sdcard/data/best.kmodel"
labels = ["Aid_Kit"]
model_input_size = [640, 640]
count = 0

try:
    # 初始化摄像头和视频输出
    sensor = Sensor(id=2, width=1280, height=720, fps=30)
    sensor.reset()

    # 给人看的通路1
    sensor.set_framesize(framesize=Sensor.VGA, chn=CAM_CHN_ID_0)
    sensor.set_pixformat(Sensor.RGB888, chn=CAM_CHN_ID_0)
    # 给ai看的通路2
    sensor.set_framesize(chn=CAM_CHN_ID_1, width=640, height=640)
    sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_1)

    Display.init(Display.VIRT, width=640, height=480, fps=30, to_ide=True)
    MediaManager.init()
    sensor.run()

    print("摄像头和显示初始化成功")

    # 初始化YOLO
    yolo = YOLOv8(
        task_type="detect",
        mode="video",
        kmodel_path=kmodel_path,
        labels=labels,
        rgb888p_size=[640, 640],
        model_input_size=[640, 640],
        conf_thresh=0.3,
        nms_thresh=0.45
    )

    print("模型加载成功")

    # 输出视频流
    while True:
        os.exitpoint()  # 检查中断点

        frame_display = sensor.snapshot(chn=CAM_CHN_ID_0)
        frame_yolo = sensor.snapshot(chn=CAM_CHN_ID_1)

        # 使用yolo进行detect并绘制到frame_display上
        results = yolo.run(frame_yolo)
        if results:
            yolo.draw(frame_display, results)

        Display.show_image(frame_display)

        # 释放内存
        frame_yolo = None
        frame_display = None

        count += 1
        if count > 60:
            print("程序运行中")
            gc.collect()
            count = 0

except KeyboardInterrupt as e:
    print(f"用户停止：{e}")
except BaseException as e:
    print(f"异常：{e}")
finally:
    if "yolo" in locals():
        yolo.deinit()
    if isinstance(sensor, Sensor):
        sensor.stop()
    Display.deinit()
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    MediaManager.deinit()