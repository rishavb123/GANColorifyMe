import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import cv2
import time

import numpy as np
import tensorflow as tf

from dataset.preproccesing import normalize

from gan.model_path import gen_path
from face_matcher.model_path import path

capture = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('../dataset/haarcascade_frontalface_default.xml')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

generator = tf.keras.models.load_model(gen_path.replace('.', '../gan'))
matcher = tf.keras.models.load_model(path)

while True:
    cur_time = time.time()
    _, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray)
    if len(faces) > 0:
        faces = list(faces)
        faces.sort(key=lambda rect: -rect[2] * rect[3])
        x, y, w, h = faces[0]

        inp_img = img[y: y + h, x: x + w]
        inp_img = cv2.resize(inp_img, (28, 28))
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
        inp_img = normalize(inp_img)
        inp_img = np.reshape(inp_img, (1, 28, 28, 1))

        out_img = generator(matcher(inp_img))
        out_img = np.reshape(out_img, (28, 28))
        out_img = normalize(out_img, input_range=(-1, 1), output_range=(0, 255))
        out_img = cv2.resize(out_img, (w, h))
        out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
        img[y: y + h, x: x + w] = out_img

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
capture.release()
