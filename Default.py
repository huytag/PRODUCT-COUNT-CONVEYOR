import cv2
from V_02 import CountProducts as cps;

URL_DETECT = './Files/datas/videoMi3Mien.mp4'

if __name__ == '__main__':
    countPros = cps.CountProducts()
    countPros.fun_initial_yolov3()

    countPros.fun_set_LEFT_RIGHT_TOP_BOTTOM_REGION(
        left_region= 130,
        right_region= 440,
        top_region= 120,
        bottom_region= 300
    )
    countPros.fun_detect_logo_digitech_video_and_count(URL_DETECT, fps= 30, skip_frame= 1, reduce_size= 1)
    