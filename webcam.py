import cv2
import time

import numpy as np
import tensorflow as tf

from cv2utils.camera import make_camera_with_args

from dataset.preproccesing import normalize

from gan.model_path import gen_path
from colorify.model_path import path

rand_start = False
stability = 10
r = 0.05

classifier = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')

generator = tf.keras.models.load_model(gen_path.replace('.', './gan'))
colorifier = tf.keras.models.load_model(path.replace('.', './colorify'))

noise = tf.random.normal([1, 100]) if rand_start else np.load('./gan/inputs/best_noise_input.npy')
count = 0
approach = tf.random.normal([1, 100])

def preprocess(raw, frames):
    global noise
    global count
    global approach
    img = frames[0]
    t0 = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, _, confidence = classifier.detectMultiScale3(gray, outputRejectLevels=True, minSize=(25, 25), maxSize=(80, 80))
    max_ind = -1
    max_metric = 0

    max_ind_2 = -1

    for i in range(len(faces)):
        metric = confidence[i]
        if metric > max_metric:
            max_metric = metric
            max_ind_2 = max_ind
            max_ind = i
    for i in [max_ind, max_ind_2]:
        if i < 0:
            continue
        x, y, w, h = faces[i]
        noise = (1 - r) * noise + r * approach
        count += 1
        if count >= stability:
            count = 0
            approach = tf.random.normal([1, 100])
        # t1 = time.time()
        gen_img = colorifier(generator(noise, training=False), training=False)
        # t2 = time.time()
        gen_img = normalize(gen_img, input_range=(-1, 1), output_range=(0, 255))
        # t3 = time.time()
        gen_img = np.reshape(gen_img, (28,28,3))
        gen_img = cv2.resize(gen_img, (w, h))
        img[y: y + h, x: x + w] = gen_img
        # t4 = time.time()
        # print(f"Find faces: {round((t1 - t0) * 1000)} ms; Generate Face: {round((t2 - t1) * 1000)} ms; Normalize: {round((t3 - t2) * 1000)} ms; Replace Image: {round((t4 - t3) * 1000)} ms; Image Shape: {img.shape}", "                                         ", end="\r")
    return img

camera, args = make_camera_with_args(log=False)

camera.make_virtual_webcam(
    preprocess=preprocess,
    prepare=None
)