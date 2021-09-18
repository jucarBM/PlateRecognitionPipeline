import torch
import cv2
import os
import boto3
import datetime


class VideoCaptured:
    """'
       Class that can be used to detect real time objects using yolo5.
       Use play_video or stream_video to start operations.
    """
    def __init__(self, cap, testname, modelType, stream=False, pad=50):
        self.model = torch.hub.load('ultralytics/yolov5',
                                    modelType,
                                    pretrained=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_conf()
        self.cap = cap
        self.pad = pad
        self.testname = testname
        self.stream = stream
        if not self.stream:
            self.frames = self.get_all_frames()

    def model_conf(self, conf=0.5, iou=0.45, classes=None):
        self.model.conf = conf
        self.model.iou = iou
        self.model.classes = classes
        self.model.to(self.device)

    def get_all_frames(self):
        frames = []
        ret, frame = self.cap.read()
        while ret:
            frames.append(frame)
            ret, frame = self.cap.read()
        self.cap.release()
        return frames

    def score_frame(self, frame):
        results = self.model(frame, size=640)
        labels = results.xyxyn[0][:, -1].numpy()
        labels_string = results.pandas().xyxyn[0]['name'].to_list()
        cords = results.xyxyn[0][:, :-1].numpy()
        return labels, labels_string, cords

    def square_operations(self, results, frame, mode):
        labels, labels_string, cord = results
        # print(labels_string)
        # print(cord)

        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        cuts = []

        for i in range(n):
            row = cord[i]
            # If score is less than 0.2 we avoid making a prediction.
            x1 = int(row[0] * x_shape)
            y1 = int(row[1] * y_shape)
            x2 = int(row[2] * x_shape)
            y2 = int(row[3] * y_shape)

            if mode == 'draw':
                bgr = (0, 255, 0)  # color of the box
                label_font = cv2.FONT_HERSHEY_SIMPLEX  # Font for the label.
                cv2.rectangle(frame,
                              (x1, y1), (x2, y2),
                              bgr, 2)  # Plot the boxes
                cv2.putText(frame, labels_string[i] + ' (' + str(row[4]) + ')',
                            (x1, y1), label_font, 1, 255)
            else:
                if y1 - self.pad < 0:
                    y1 = y1
                else:
                    y1 = y1 - self.pad

                if y2 + self.pad > y_shape:
                    y2 = y2
                else:
                    y2 = y2 + self.pad

                if x1 - self.pad < 0:
                    x1 = x1
                else:
                    x1 = x1 - self.pad

                if x2 + self.pad > x_shape:
                    x2 = x2
                else:
                    x2 = x2 + self.pad

                cuts.append(frame[y1:y2, x1:x2])

        if mode == 'cut':
            return cuts
        else:
            return frame

    def stream_video(self, recognize, mode):
        if not self.cap.isOpened():
            print("Cannot open camera")
            return None
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            print(ret)
            print('-----')
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Display the resulting frame
            if recognize:
                results = self.score_frame(frame)
                frame = self.square_operations(results, frame, mode)
            if mode == 'cut':
                cv2.imshow('frame', frame[0])
            else:
                cv2.imshow('frame', frame)
            print(results)
            if cv2.waitKey(1) == ord('q'):
                break
        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()

    def play_video(self, recognize, mode, save=False):
        count = 0
        path = 'results/' + self.testname
        if save and not os.path.isdir(path):
            os.makedirs(path)

        for frame in self.frames:
            if recognize:
                frame_rec = self.square_operations(self.score_frame(frame), frame, mode)
                if frame_rec is not None:
                    if mode == 'cut':
                        for cut_frame in frame_rec:
                            cv2.imshow('frame', cut_frame)
                            if count == 211:
                                print('hola')
                            if save:
                                filename = path + '/' + str(count) + '.jpg'
                                cv2.imwrite(filename, cut_frame)
                                count += 1
                    else:
                        cv2.imshow('frame', frame_rec)

                    if cv2.waitKey(1) == ord('q'):
                        break
                else:
                    cv2.imshow('frame', frame)
            else:
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            print(count)
        cv2.destroyAllWindows()


class FramesSended:
    """"
        Class that uses yolov5 toi detect specific objects and send frames to S3 buckets.
    """
    def __init__(self, cap, rbpiID, modelType, bucketName):
        """"
            Inputs:
                Cap: capture object from opencv, it have to be a stream video
                sessionName: Name of the session
                modelType: Model used in yolov5. i.e. = yolov5s, yolov5l
        """
        self.model = torch.hub.load('ultralytics/yolov5',
                                    modelType,
                                    pretrained=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_conf()
        self.cap = cap
        self.rbpiID = rbpiID
        self.bucketName = bucketName
        # Creating the high level object oriented interface
        self.resource = boto3.resource(
            's3',
            aws_access_key_id='AKIAZC5KVF2CCPQ342Z4',
            aws_secret_access_key='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',
            region_name='us-west-2'
        )

    def model_conf(self, conf=0.5, iou=0.5, classes=None):
        """"
            Configuration of the yolo5 model. Ref: https://github.com/ultralytics/yolov5/issues/36
            model.conf = confidence threshold (0-1), default 0.5
            model.iou = NMS IoU threshold (0-1), default, 0.5
            model.classes = (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
        """
        self.model.conf = conf
        self.model.iou = iou
        self.model.classes = classes
        self.model.to(self.device)

    def score_frame(self, frame):
        """"
            Score frame and detect objects.
            Inputs
                frame: frame to be analized
            Outputs
                labels: class number code
                labels_string: string of the class
                cords: cordenates of the square where the object is
        """
        results = self.model(frame, size=640)
        # labels = results.xyxyn[0][:, -1].numpy()
        labels_string = results.pandas().xyxyn[0]['name'].to_list()
        # cords = results.xyxyn[0][:, :-1].numpy()
        return labels_string

    def start_process(self):
        """"
            Starts sending images detected to a bukets s3.
        """
        if not self.cap.isOpened():
            print("Cannot open camera")
            return None
        while True:
            # Capture frame-by-frame
            # datetime.datetime.now()
            # frameRate = self.cap.get(5)  # frame rate
            ret, frame = self.cap.read()
            frameId = self.cap.get(1)  # current frame number
            # print(frameId)
            # print(ret)
            # print('-----')

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Our operations on the frame come
            ###########################################
            date = datetime.datetime.now()
            image_string = cv2.imencode('.jpg', frame)[1].tostring()
            labels = self.score_frame(frame)
            # print(labels)
            if labels:
                fileName = date.strftime('%Y%m%d') + date.strftime('_%H%M%S_') + str(labels[0]) \
                           + '_' + str(frameId) + '.jpg'
                # 20210910_223423_car_323773.jpg
                # print(fileName)
                destinationPath = str(self.rbpiID) + '/' + date.strftime('%Y') + '/' + date.strftime('%m') + \
                                  '/' + date.strftime('%d') + '/' + fileName
                # print(destinationPath)
                self.resource.Bucket(self.bucketName).put_object(Key=destinationPath,
                                                                 Body=image_string)
                # print('hay una imagen')
            ############################################
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) == ord('q'):
            #     break

        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()

