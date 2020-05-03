import os

import numpy as np
import cv2

def load_data(image_dir):
    faces = []
    gray = []

    for fname in os.listdir(image_dir + '/color/'):
        faces.append(cv2.imread(image_dir + "/color/" + fname))
        g = cv2.imread(image_dir + "/grayscale/" + fname)
        gcopy = []
        for i in range(len(g)):
            gcopy.append([])
            for j in range(len(g[i])):
                gcopy[i].append(g[i][j][0])
        gray.append(gcopy)

    return np.array(faces, float), np.array(gray, float)

def load_color_data(image_dir):
    faces = []

    for fname in os.listdir(image_dir + '/color/'):
        faces.append(cv2.imread(image_dir + "/color/" + fname))

    return np.array(faces, float)

def load_gray_data(image_dir):
    gray = []

    for fname in os.listdir(image_dir + '/color/'):
        g = cv2.imread(image_dir + "/grayscale/" + fname)
        gcopy = []
        for i in range(len(g)):
            gcopy.append([])
            for j in range(len(g[i])):
                gcopy[i].append(g[i][j][0])
        gray.append(gcopy)

    return np.array(gray, float)

def normalize(faces, output_range=(-1, 1), input_range=(0, 255)):
    return (faces - input_range[0]) * (output_range[1] - output_range[0]) / (input_range[1] - input_range[0]) + output_range[0]