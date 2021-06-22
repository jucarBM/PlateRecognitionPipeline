# Main script for test
import cv2
import torch
from imageProcessing.utilities import VideoCaptured
import numpy as np

if __name__ == '__main__':
    print('Starting ...')

    # variables
    pad = 50

    # Model configuration
    modelType = 'yolov5m6'

    # Video
    cap = cv2.VideoCapture('videos/doscarros.mp4')
    # cap = cv2.VideoCapture(2)
    video = VideoCaptured(cap, 'doscarros', modelType, stream=False)
    video.model_conf(classes=[[2, 3, 5, 7]])
    # video.stream_video(recognize=True, mode='cut')
    video.play_video(recognize=True, mode='cut', save=True)


