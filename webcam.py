import cv2

import numpy as np
import tensorflow as tf

from cv2utils.camera import make_camera_with_args

from dataset.preproccesing import normalize

from gan.model_path import gen_path
from colorify.model_path import path

rand_start = False
stability = 10
r = 0.1

classifier = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')

generator = tf.keras.models.load_model(gen_path.replace('.', './gan'))
colorifier = tf.keras.models.load_model(path.replace('.', './colorify'))

noise = tf.random.normal([1, 100]) if rand_start else np.load('./gan/inputs/best_noise_input.npy')
count = 0
approach = tf.random.normal([1, 100])

def preprocess(frames):
    global noise
    global count
    global approach
    img = frames[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray)
    for face in faces:
        x, y, w, h = face

        noise = (1 - r) * noise + r * approach
        count += 1
        if count >= stability:
            count = 0
            approach = tf.random.normal([1, 100])

        gen_img = colorifier(generator(noise, training=False), training=False)
        gen_img = normalize(gen_img, input_range=(-1, 1), output_range=(0, 255))
        gen_img = np.reshape(gen_img, (28,28,3))
        gen_img = cv2.resize(gen_img, (w, h))
        img[y: y + h, x: x + w] = gen_img
    return img

camera, args = make_camera_with_args(log=False)

camera.make_virtual_webcam(
    preprocess=preprocess
)