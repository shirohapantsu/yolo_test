"""
yolo_in_k230 的 Docstring
-------------------------------

在k230上部署yolov8实现detect医疗包，画面输出至ide

author:shiroha_pantsu
date:2026-01-21
"""

import os, sys, gc
from libs.PipeLine import PipeLine
from Libs.YOLO import YOLOv8
from libs.Utils import *
import ulab.numpy as np
import image

# 配置模型参数
kmodel_path = "sdcard/data/best.kmodel"
labels = ["Aid_Kit"]
model_input_size = [640, 640]
count = 0

display_mode="lcd"
rgb888p_size=[640,640]
confidence_threshold = 0.5
nms_threshold=0.45

try:
    #初始化pipeline
    pl=PipeLine(
        rgb888p_size=rgb888p_size,
        display_mode=display_mode
    )
    pl.create()
    display_size=pl.get_display_size()
    print("初始化pipeline成功")

    # 初始化YOLO
    yolo = YOLOv8(
        task_type="detect",
        mode="video",
        kmodel_path=kmodel_path,
        labels=labels,
        rgb888p_size=rgb888p_size,
        model_input_size=model_input_size,
        display_size=display_size,
        conf_thresh=confidence_threshold,
        nms_thresh=nms_threshold
    )
    yolo.config_preprocess()

    print("模型加载成功")

    # 主循环
    while True:
        os.exitpoint()  # 检查中断点

        img=pl.get_frame()
        # 使用yolo进行detect并绘制到frame_display上
        results = yolo.run(img)
        if results:
            yolo.draw_result(results,pl.osd_img)

        pl.show_image()

        # 释放内存
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
    if "pl" in locals():
        pl.destroy()
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)

