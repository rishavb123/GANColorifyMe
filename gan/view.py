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

capture = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('../dataset/haarcascade_frontalface_default.xml')

generator = tf.keras.models.load_model('./models/1588550609579/generator')

noise = tf.random.normal([1, 100])

while True:
    cur_time = time.time()
    _, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray)
    if len(faces) > 0:
        faces = list(faces)
        faces.sort(key=lambda rect: -rect[2] * rect[3])
        x, y, w, h = faces[0]

        noise = 0.9 * noise + 0.1 * tf.random.normal([1, 100])

        gen_img = generator(noise, training=False)
        gen_img = normalize(gen_img, input_range=(-1, 1), output_range=(0, 255))
        gen_img = np.reshape(gen_img, (28,28))
        gen_img = cv2.resize(cv2.cvtColor(gen_img, cv2.COLOR_GRAY2BGR), (w, h))
        img[y: y + h, x: x + h] = gen_img

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
capture.release()
