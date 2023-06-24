from ultralytics import YOLO
import cv2
import time
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device
import yaml
from random import randint
from ultralytics.SORT import *
from tensorflow import keras
import tensorflow as tf
from keras.models import load_model

yolo_model = YOLO("yolov8s-pose.pt")

model = load_model("model/pose_estimation.h5",compile= False)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()])

l=[]

dic = {0: "Sitting", 1:"Standing", 2: "Sitting and Raising Hand", 3: "Standing and Raising Hand"}
for c in range(10):
    try:
        ret, frame = cv2.VideoCapture(c).read()

        if ret: 
            l.append(c)
    except:
        pass
print(l)

start = time.time()
cap = cv2.VideoCapture(l[0])

rand_color_list = np.random.rand(20, 3) * 255

while cap.isOpened():
    res = []
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # print("fps: " + str(round(1 / (time.time() - start), 2)))
    start = time.time()
    # frame2 = np.copy(frame)

    results = yolo_model.predict(source=frame, conf=0.7, show=True)[0]
  
    # print(results.boxes)
    # print(results.keypoints)
    # print(results.keypoints.shape)
    if results.boxes:
        flattened_data = results.keypoints.flatten()
        reshaped_tensor = np.array(flattened_data).reshape(1, -1)
        print(reshaped_tensor.shape)

        p = model.predict(reshaped_tensor)
        p_index = np.argmax(p, axis = 1)
        print(dic[int(p_index)])

    # cv2.imshow("frame", frame2)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
