from ultralytics import YOLO
import cv2
import time
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device
import yaml
from random import randint
from tensorflow import keras
import tensorflow as tf
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracker.byte_tracker import STrack
from BYTETrackerArgs import BYTETrackerArgs
import argparse
from typing import Tuple, Optional, List, Dict, Any
from onemetric.cv.utils.iou import box_iou_batch

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        print(e)

# run with pytorch model
# model = YOLO("weights/yolov8s-pose.pt", task="pose")

# run with TensorRT Model
model = YOLO("weights/yolov8l-pose.engine", task="pose")

# print("Loading Keras Model")
# try:
#     keras_model = keras.models.load_model(
#         "weights/first_weight.h5", compile=False)
#     pred_keras = True
#     print("DONE")
# except:
#     print("Error while loading keras model")
#     pred_keras = False
#     keras_model = ""

KEYPOINTS_CONF = 0.7
YOLO_CONF = 0.7

tracker = BYTETracker(BYTETrackerArgs(), frame_rate = 30)

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

def process_keypoints(keypoints, conf, frame_width, frame_height, origin=(0, 0)):
    kpts = np.copy(keypoints)
    kpts[:, 0] = (kpts[:, 0] - origin[0]) / frame_width
    kpts[:, 1] = (kpts[:, 1] - origin[1]) / frame_height

    kpts[:, :-1][kpts[:, 2] < conf] = [-1, -1]
    return np.round(kpts[:, :-1].flatten(), 4)

def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

def match_detections_with_tracks(
    detections: np.ndarray,
    tracks: List[STrack],
) -> np.ndarray:
    detection_boxes = detections[:, :4].astype(float)
    tracks_boxes = tracks2boxes(tracks=tracks)
    if np.size(tracks_boxes) == 0 or np.size(detection_boxes) == 0:
        return np.array([])
    else:
        iou = box_iou_batch(tracks_boxes, detection_boxes)
        track2detection = np.argmax(iou, axis=1)

        detections = np.insert(detection_boxes, 4, 0, axis = 1)

        for tracker_index, detection_index in enumerate(track2detection):
            if iou[tracker_index, detection_index] != 0:
                # print("detections[detection_index]:", detections[detection_index], type(detections[detection_index]))
                # print("tracks[tracker_index].track_id:", tracks[tracker_index].track_id, type(tracks[tracker_index].track_id))
                detections[detection_index][4] = tracks[tracker_index].track_id

    return detections


start = time.time()
cap = cv2.VideoCapture(list_available_cam(10))
# cap = cv2.VideoCapture('data/1.mp4')

FRAME_WIDTH = cap.get(3)
FRAME_HEIGHT = cap.get(4)

rand_color_list = np.random.rand(20, 3) * 255

while cap.isOpened():
    res = []
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    results = model.predict(source=frame, conf=YOLO_CONF,
                          verbose=False)[0]
    kpts = results.keypoints.cpu().numpy()
    boxes = results.boxes.data.cpu().numpy()
    # print(kpts)
    output = {}
    for person_kpts, person_box in zip(kpts, boxes):
        # print(person_box)

        x1, y1, x2, y2 = person_box[:4]
        #person_id = int(person_box[4])

        processed_kpts = process_keypoints(
            person_kpts, KEYPOINTS_CONF, FRAME_WIDTH, FRAME_HEIGHT, (x1, y1))
        
        # pred_pose = np.argmax(keras_model.predict(
        #     processed_kpts.reshape((1, 34)), verbose=0), axis=1)

        dets = np.array(person_box[:5])[np.newaxis,:]
        info_imgs = (FRAME_HEIGHT, FRAME_WIDTH)
        img_size = (FRAME_HEIGHT, FRAME_WIDTH)

        targets = tracker.update(dets,info_imgs,img_size)
        tracks_boxes = tracks2boxes(tracks=targets)
        matched_detections = match_detections_with_tracks(person_box[:4][np.newaxis,:],targets)
        if np.size(matched_detections) != 0:
            person_id = matched_detections[0][4]
            output[int(person_id)] = {"name": 0, "bbox": matched_detections[0][:4]}

        #print("boxes:", person_box[:4][np.newaxis,:])
        #print("track_boxes:", tracks_boxes) #predicted by bytetrack
        #print("targets:", targets)
        #print("matched:", matched_detections)
        cv2.putText(frame, str(person_id), (int(x1), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), (255, 0, 0), 2)

        # Draw points
        for i, pt in enumerate(person_kpts):
            x, y, p = pt
            if p >= KEYPOINTS_CONF:
                cv2.putText(frame, str(i), (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
    print(output)

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    start = time.time()
    # frame2 = np.copy(frame)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
