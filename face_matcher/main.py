import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import datetime
import time

import tensorflow as tf
import numpy as np


from face_matcher.config import *
from face_matcher.make_model import make_model
from gan.model_path import gen_path
from gan.config import NOISE_DIM


model = make_model()

inputs = np.load(inputs_data_file)
outputs = np.load(outputs_data_file)

inputs = tf.data.Dataset.from_tensor_slices(inputs)
outputs = tf.data.Dataset.from_tensor_slices(outputs)
dataset = tf.data.Dataset.zip((inputs, outputs)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

model.compile(optimizer='adam', loss='mse')
model.fit(dataset, epochs=EPOCHS)

s = str(int(time.time() * 1000))
os.mkdir('./models/' + s)
model.save('./models/' + s)