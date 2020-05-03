import os

import time
import datetime
import numpy as np
import cv2

def load_data(image_dir, progress_bar_width=80, progress_char='+', empty_char='-', should_log=True, load_color=True, load_gray=True):
    faces = []
    gray = []

    ls = os.listdir(image_dir + '/color/')
    l = len(ls)

    def log(*args, **kwargs):
        if should_log:
            print(*args, **kwargs)

    log('Loading data from', '"' + image_dir + '"', '. . .')

    cur_time = time.time()

    for i in range(l):
        w = i * progress_bar_width // l
        s = int(time.time() - cur_time)
        log('[' + progress_char * w + empty_char * (progress_bar_width - w) + ']', i, '/', l, datetime.timedelta(seconds=s), end='\r')

        fname = ls[i]
        if load_color:
            faces.append(cv2.imread(image_dir + "/color/" + fname))
        if load_gray:
            g = cv2.imread(image_dir + "/grayscale/" + fname)
            gcopy = []
            for i in range(len(g)):
                gcopy.append([])
                for j in range(len(g[i])):
                    gcopy[i].append([g[i][j][0]])
            gray.append(gcopy)
    log('[' + progress_char * progress_bar_width + ']', l, '/', l)
    log('Finished loading data - Took', int(time.time() - cur_time), 'seconds')
    if load_color and load_gray:
        return np.array(faces, float), np.array(gray, float)
    elif load_color:
        return np.array(faces, float)
    elif load_gray:
        return np.array(gray, float)
    return np.array([], float)

def normalize(faces, output_range=(-1, 1), input_range=(0, 255)):
    return (faces - input_range[0]) * (output_range[1] - output_range[0]) / (input_range[1] - input_range[0]) + output_range[0]