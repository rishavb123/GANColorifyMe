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

nx, ny = 2, 4
a, b, h = -1, 1, 0.02
inp = np.arange(a, b + h, h).reshape(-1, 1)
gaussian_vec_x = np.exp(- 4 * inp ** (2 * nx)).T
gaussian_vec_y = np.exp(- 4 * inp ** (2 * ny))
gaussian_matrix = gaussian_vec_y @ gaussian_vec_x
gaussian_tensor = np.stack((gaussian_matrix, gaussian_matrix, gaussian_matrix), axis=2)

def preprocess(raw, frames):
    global noise
    global count
    global approach
    img = frames[0]
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
        gen_img = colorifier(generator(noise, training=False), training=False)
        gen_img = normalize(gen_img, input_range=(-1, 1), output_range=(0, 255))
        gen_img = np.reshape(gen_img, (28,28,3))
        gen_img = cv2.resize(gen_img, (w, h))
        weights = cv2.resize(gaussian_tensor, (w, h))
        img[y: y + h, x: x + w] = gen_img * weights + (1 - weights) * img[y: y + h, x: x + w]
    return img

camera, args = make_camera_with_args(log=False)

camera.make_virtual_webcam(
    preprocess=preprocess,
    prepare=None
)