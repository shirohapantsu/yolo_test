import os,time,sys
from media.display import *
from media.media import *
from media.sensor import *

try:
    # 初始化摄像头和视频输出
    sensor = Sensor(id=2,width=1280,height=720,fps=30)
    sensor.reset()
    sensor.set_framesize(framesize=Sensor.HD,chn=CAM_CHN_ID_0)
    sensor.set_pixformat(Sensor.RGB888,chn=CAM_CHN_ID_0)

    Display.init(Display.VIRT,width=1280,height=720,fps=30,to_ide=True)
    MediaManager.init()
    sensor.run()


    #输出视频流
    while True:
        os.exitpoint() #检查中断点
        frame=sensor.snapshot(chn=CAM_CHN_ID_0)
        Display.show_image(frame)

except KeyboardInterrupt as e:
    print(f"用户停止：{e}")
except BaseException as e:
    print(f"异常：{e}")
finally:
    if isinstance(sensor,Sensor):
        sensor.stop()
    Display.deinit()
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    MediaManager.deinit()
