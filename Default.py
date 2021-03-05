import cv2
from V_02 import CountProducts as cps
import json
import os

URL_FILE_CONFIG = './project.config.json'

CONFIG = {
    "URL_LOAD_VIDEO": "./Files/Videos/01_Classess_3Mien.mp4",
    "WEIGHT_NAME": "./Files/01_Classess_3Mien/01_Classess_3Mien.weights",
    "CONF_NAME": "./Files/01_Classess_3Mien/01_Classess_3Mien.cfg",
    "CLASSES_NAMES": "./Files/01_Classess_3Mien/01_Classess_3Mien.txt",
    "CORLOR": [51, 255, 102],
    "LEFT_REGION": 130,
    "RIGHT_REGION": 440,
    "TOP_REGION": 120,
    "BOTTOM_REGION": 300,
    "COPYRIGHT": "Digitech Solutions",
    "COPYRIGHT_COLOR": [51, 255, 102],
    "COPYRIGHT_FONT_SIZE": 0.5
}


def fun_createConfigFile():
    with open(URL_FILE_CONFIG, 'w') as f:
        json.dump(CONFIG, f)
    print('----------------------------------------------------------------------')
    print('MAYBE THE APPLICATION IS NOT WORKING CORRECT, PLEASE EDIT CONFIG FILE!')
    print('----------------------------------------------------------------------')


def fun_loadConfigFile():
    with open(URL_FILE_CONFIG, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    # Create config file if not exists
    if not os.path.exists(URL_FILE_CONFIG):
        fun_createConfigFile()

    # Load config file
    CONFIG = fun_loadConfigFile()

    # Initial App
    countPros = cps.CountProducts(CONFIG)
    
    # Start App
    countPros.fun_startVideoAndCountObject(fps= 1, skip_frame= 1, reduce_size= 1)