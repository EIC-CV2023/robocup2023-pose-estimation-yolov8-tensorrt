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

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        print(e)

model = YOLO("weights/yolov8s-pose.pt", task="pose")

print("Loading Keras Model")
try:
    keras_model = keras.models.load_model(
        "weights/first_weight.h5", compile=False)
    pred_keras = True
    print("DONE")
except:
    print("Error while loading keras model")
    pred_keras = False
    keras_model = ""


YOLO_CONF = 0.7
KEYPOINTS_CONF = 0.7


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

    results = model.track(source=frame, conf=YOLO_CONF,
                          show=True, verbose=False, persist=True)[0]
    kpts = results.keypoints.cpu().numpy()
    boxes = results.boxes.data.cpu().numpy()
    # print(boxes)
    # print(kpts)

    for person_kpts, person_box in zip(kpts, boxes):
        # print(person_box)
        x1, y1, x2, y2 = person_box[:4]
        person_id = int(person_box[4])
        # print(person_id)

        # print(x1,y1)

        processed_kpts = process_keypoints(
            person_kpts, KEYPOINTS_CONF, FRAME_WIDTH, FRAME_HEIGHT, (x1, y1))
        # print(processed_kpts)

        pred_pose = np.argmax(keras_model.predict(
            processed_kpts.reshape((1, 34)), verbose=0), axis=1)
        print(pred_pose[0])

        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), (255, 0, 0), 2)

        # Draw points
        for i, pt in enumerate(person_kpts):
            x, y, p = pt
            if p >= KEYPOINTS_CONF:
                cv2.putText(frame, str(i), (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # print("fps: " + str(round(1 / (time.time() - start), 2)))
    start = time.time()
    # frame2 = np.copy(frame)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
