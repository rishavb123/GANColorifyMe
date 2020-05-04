import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import time

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.preproccesing import normalize

from colorify.model_path import path
from colorify.config import color_data_file, gray_data_file

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)


colorifier = tf.keras.models.load_model(path)

ind = 0
color = np.load(color_data_file)[ind : ind + 1]
gray = np.load(gray_data_file)[ind : ind + 1]

plt.imshow(cv2.cvtColor(color[0], cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(gray[0, :, :, 0], cmap='gray')
plt.show()
img = np.array(colorifier(gray)[0, :, :, :])
img = np.array([[[c[2], c[1], c[0]] for c in r] for r in img])
plt.imshow(img)
plt.show()