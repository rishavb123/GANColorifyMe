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
from gan.model_path import gen_path
from gan.config import NOISE_DIM


progress_bar_width = 100
filled = '+'
empty = '-'

inputs = np.zeros((NUM_SAMPLES, 28, 28, 1))
outputs = np.zeros((NUM_SAMPLES, 1, NOISE_DIM))

generator = tf.keras.models.load_model(gen_path.replace('.', '../gan'))

print('Generating Data . . .')
start_time = time.time()
for i in range(NUM_SAMPLES):
    noise = tf.random.normal([1, NOISE_DIM])
    outputs[i] = noise
    img = generator(noise, training=False)
    inputs[i] = img
    p = (i * progress_bar_width // NUM_SAMPLES)
    s = int(time.time() - start_time)
    print('[' + filled * p + empty * (progress_bar_width - p) + ']', i, '/', NUM_SAMPLES, datetime.timedelta(seconds=s), end='\r')
print('[' + filled * progress_bar_width+ ']', NUM_SAMPLES, '/', NUM_SAMPLES, datetime.timedelta(seconds=s))
print('Finished Generating Data in', int(time.time() - start_time), 'seconds')

print('Saving Data . . .')
start_time = time.time()
np.save(inputs_data_file, inputs)
np.save(outputs_data_file, outputs)
print('Finished Saving Data in', int(time.time() - start_time), 'seconds')