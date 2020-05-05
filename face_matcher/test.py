import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import datetime
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


from face_matcher.config import *
from face_matcher.model_path import path
from gan.model_path import gen_path
from dataset.preproccesing import normalize

generator = tf.keras.models.load_model(gen_path.replace('.', '../gan'))
matcher = tf.keras.models.load_model(path)

noise = tf.random.normal([1, 100])
img = np.array(generator(noise, training=False))
matched_noise = matcher(img, training=False)
print(np.linalg.norm(noise - matched_noise))

matched = np.array(generator(matched_noise))

plt.imshow(img[0, :, :, 0], cmap='gray')
plt.show()

plt.imshow(matched[0, :, :, 0], cmap='gray')
plt.show()
