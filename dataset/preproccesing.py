import os

import numpy as np
import cv2

def load_data():
    faces = []
    gray = []

    for fname in os.listdir('./images/color/'):
        faces.append(cv2.imread("./images/color/" + fname))
        g = cv2.imread("./images/grayscale/" + fname)
        gcopy = []
        for i in range(len(g)):
            gcopy.append([])
            for j in range(len(g[i])):
                gcopy[i].append(g[i][j][0])
        gray.append(gcopy)

    return np.array(faces, float), np.array(gray, float)

def load_color_data():
    faces = []

    for fname in os.listdir('./images/color/'):
        faces.append(cv2.imread("./images/color/" + fname))

    return np.array(faces, float)

def load_gray_data():
    gray = []

    for fname in os.listdir('./images/color/'):
        g = cv2.imread("./images/grayscale/" + fname)
        gcopy = []
        for i in range(len(g)):
            gcopy.append([])
            for j in range(len(g[i])):
                gcopy[i].append(g[i][j][0])
        gray.append(gcopy)

    return np.array(gray, float)

def normalize(faces, output_range=(-1, 1), input_range=(0, 255)):
    return (faces - input_range[0]) * (output_range[1] - output_range[0]) / (input_range[1] - input_range[0]) + output_range[0]