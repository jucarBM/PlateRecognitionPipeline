# Main script for test
import cv2
from imageProcessing.utilities import VideoCaptured, FramesSended

'''
set AWS_SECRET_ACCESS_KEY=SvinHqNhKFxcTAhLe6wsNzqHjQ6zngiuWCYamH5h
set AWS_ACCESS_KEY_ID=AKIAZC5KVF2CCPQ342Z4
set AWS_DEFAULT_REGION=us-west-2
'''

if __name__ == '__main__':
    print('Starting ...')

    # variables
    pad = 50
    rbpiID = 'rbpi_0'
    bucketName = 'framesfromcameras'

    # Model configuration
    modelType = 'yolov5s6'
    # Video
    # cap = cv2.VideoCapture("rtsp://192.168.1.109:554/11")
    # cap = cv2.VideoCapture("rtsp://admin:123@169.254.249.184/ch0_0.264")

    cap = cv2.VideoCapture(0)

    video = FramesSended(cap, rbpiID, modelType, bucketName)

    video.start_process()
