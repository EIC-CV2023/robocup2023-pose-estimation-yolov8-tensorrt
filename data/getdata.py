from ultralytics import YOLO
import cv2
import time
import numpy as np
import yaml
from random import randint
import pandas as pd
import os
import csv

model = YOLO("weights/yolov8s-pose.pt")
YOLO_CONF = 0.7
KEYPOINTS_CONF = 0.7

label = 3
VID_PATH = "data/3.mp4"


def list_available_cam(max_n):
    list_cam = []
    for n in range(max_n):
        cap = cv2.VideoCapture(n)
        ret, _ = cap.read()

        if ret:
            list_cam.append(n)
        cap.release()
    
    if len(list_cam) == 1:
        return list_cam[0]
    else:
        print(list_cam)
        return int(input("Cam index: "))
    

def process_keypoints(keypoints, conf, frame_width, frame_height, origin = (0,0)):
    kpts = np.copy(keypoints)
    kpts[:,0] = (kpts[:,0] - origin[0]) / frame_width
    kpts[:,1] = (kpts[:,1] - origin[1]) / frame_height

    kpts[:,:-1][kpts[:,2] < conf] = [-1,-1]
    return np.round(kpts[:,:-1].flatten(),4)


def save_data_to_csv(data, label, filename):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row = data + [label]
        writer.writerow(row)


def save_data_(data, label, filename):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row = np.concatenate((data, [label]))
        writer.writerow(row)


start = time.time()
cap = cv2.VideoCapture(VID_PATH)

FRAME_WIDTH = cap.get(3)
FRAME_HEIGHT = cap.get(4)

rand_color_list = np.random.rand(20, 3) * 255


while cap.isOpened():
    res = []
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    
    start = time.time()

    results = model.predict(source=frame, conf=YOLO_CONF, show=True, verbose=False)[0]
    kpts = results.keypoints.cpu().numpy()
    boxes = results.boxes.data.cpu().numpy()
    # print(boxes)
    # print(kpts)

    for person_kpts, person_box in zip(kpts, boxes):

        x1, y1, x2, y2 = person_box[:-2]
        # print(x1,y1)

        processed_kpts = process_keypoints(person_kpts, KEYPOINTS_CONF, FRAME_WIDTH, FRAME_HEIGHT, (x1, y1))
        print(processed_kpts)

        save_data_to_csv(list(processed_kpts),label,"dataset.csv")

        # pred_pose = np.argmax(keras_model.predict(processed_kpts, verbose=0), axis=1)

        cv2.rectangle(frame, (int(x1), int(y1)),(int(x2), int(y2)), (255,0,0), 2)

        # Draw points
        for i, pt in enumerate(person_kpts):
            x, y, p = pt
            if p >= KEYPOINTS_CONF:
                cv2.putText(frame, str(i), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
