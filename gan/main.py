import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import tensorflow as tf

from dataset.preproccesing import load_data, normalize

gray = load_data('../dataset/images/', should_log=True, load_color=False)
gray = normalize(gray)
print(gray[0][0][0])