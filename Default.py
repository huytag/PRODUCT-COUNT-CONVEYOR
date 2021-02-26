import cv2
from V_02 import CountProducts as cps;

URL_DETECT = './Files/ProductsEdit.mp4'

if __name__ == '__main__':
    countPros = cps.CountProducts();
    countPros.fun_set_LEFT_RIGHT_TOP_BOTTOM_REGION(
        left_region= 10,
        right_region= 280,
        top_region= 270,
        bottom_region= 450
    )
    countPros.fun_detect_logo_digitech_video(URL_DETECT, fps= 30)
    