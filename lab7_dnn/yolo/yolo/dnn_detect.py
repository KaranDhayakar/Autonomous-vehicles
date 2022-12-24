import cv2
import time
import os
import argparse
import numpy as np

import rclpy
from rclpy.node import Node


class Dnn():
    def __init__(self,nms=0.4):

        self.modelfolder = os.path.join(os.getenv('HOME'),'av/models')
        self.datafolder = os.path.join(os.getenv('HOME'),'av/data')

        self.nms_threshold = 0.4
        self.NMS = nms

        self.COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

        self.class_names = []
        with open(os.path.join(self.modelfolder,"coco.names"), "r") as f:
                self.class_names = [cname.strip() for cname in f.readlines()]

        self.net = cv2.dnn.readNet(os.path.join(self.modelfolder,"yolov4-mish-416.weights"), os.path.join(self.modelfolder,"yolov4-mish-416.cfg"))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        self.run_time = 0


    def detect(self,frame): 
        start = time.time()
        classes, scores, boxes = self.model.detect(frame, self.nms_threshold, self.NMS)
        end = time.time()
        self.run_time = end - start
        return classes, scores, boxes

    def draw(self, frame, classes, scores, boxes):
        start_draw = time.time()
        for (classid, score, box) in zip(classes, scores, boxes): 
                color = self.COLORS[int(classid) % len(self.COLORS)]
                label = "%s : %f" % (self.class_names[classid], score)
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        end = time.time()
        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (self.run_time), (end - start_draw) * 1000)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("detections", frame)
        return


if __name__ == "__main__":
    dnn = Dnn()
    parser = argparse.ArgumentParser(description = 'input video .mp4')
    parser.add_argument('vid', help='Name your of video file? example.mp4')
    args = parser.parse_args()
    vid = os.path.join(self.datafolder,"msu_bags/"+ str(args.vid))
    vc = cv2.VideoCapture(vid)
    stop = cv2.waitKey(1)

    while stop < 1:
        if (stop == ord('q')) or (stop == ord('Q')):
            exit()
        (grabbed, frame) = vc.read()
        if not grabbed:
            exit()
        classes, scores, boxes = dnn.detect(frame)
        dnn.draw(frame, classes, scores, boxes)
        stop = cv2.waitKey(1)



