import torch
import cv2
import os


class VideoCaptured:
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

    def model_conf(self, conf=0.3, iou=0.45, classes=None):
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
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Display the resulting frame
            if recognize:
                frame = self.square_operations(self.score_frame(frame), frame, mode)
            if mode == 'cut':
                cv2.imshow('frame', frame[0])
            else:
                cv2.imshow('frame', frame)

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
