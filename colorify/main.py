import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import time
import datetime

import numpy as np
import tensorflow as tf
import cv2

from dataset.preproccesing import load_data, normalize

from colorify.make_model import make_model
from colorify.config import *

if os.path.exists(gray_data_file) and not update_data_file:
    color = np.load(color_data_file)
    gray = np.load(gray_data_file)
else:
    color, gray = load_data('../dataset/images/', should_log=True)
    color, gray = normalize(color), normalize(gray)
    np.save(color_data_file, color)
    np.save(gray_data_file, gray)

gray = tf.data.Dataset.from_tensor_slices(gray)
color = tf.data.Dataset.from_tensor_slices(color)
dataset = tf.data.Dataset.zip((gray, color)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

log_dir= ".\\logs\\fit\\"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model = make_model()
model.compile(optimizer='adam', loss='mse')
model.fit(dataset, epochs=10, callbacks=[tensorboard_callback])

s = str(int(time.time() * 1000))
os.mkdir('./models/' + s)
model.save('./models/' + s)
