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

from colorify.model_path import path

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

capture = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('../dataset/haarcascade_frontalface_default.xml')

colorifier = tf.keras.models.load_model(path)

while True:
    cur_time = time.time()
    _, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray)
    if len(faces) > 0:
        faces = list(faces)
        faces.sort(key=lambda rect: -rect[2] * rect[3])
        x, y, w, h = faces[0]

        inp_img = normalize(cv2.resize(gray[y: y + h, x: x + h], (28, 28)))
        inp_img = np.reshape(inp_img, (1, 28, 28, 1))
        gen_img = colorifier(inp_img, training=False)
        gen_img = normalize(gen_img, input_range=(-1, 1), output_range=(0, 255))
        gen_img = np.reshape(gen_img, (28,28,3))
        gen_img = cv2.resize(gen_img, (w, h))
        img[y: y + h, x: x + w] = gen_img

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
capture.release()
