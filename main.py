# Main script for test
import cv2
import boto3
import torch
from imageProcessing.utilities import VideoCaptured
import numpy as np
'''
set AWS_SECRET_ACCESS_KEY=SvinHqNhKFxcTAhLe6wsNzqHjQ6zngiuWCYamH5h
set AWS_ACCESS_KEY_ID=AKIAZC5KVF2CCPQ342Z4
set AWS_DEFAULT_REGION=us-west-2
'''

if __name__ == '__main__':
    print('Starting ...')

    # variables
    pad = 50

    # Model configuration
    modelType = 'yolov5s'
    # Video
    # cap = cv2.VideoCapture("rtsp://admin:123@172.14.10.1/ch0_0.264")
    # cap = cv2.VideoCapture("rtsp://admin:123@169.254.249.184/ch0_0.264")

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(endpoint)
    # cap = cv2.VideoCapture(2)
    video = VideoCaptured(cap, 'testCamera', modelType, stream=True)
    # video.model_conf(classes=[[2, 3, 5, 7]])
    video.stream_video(recognize=False, mode='notcut')
    # video.play_video(recognize=True, mode='cut', save=True)


