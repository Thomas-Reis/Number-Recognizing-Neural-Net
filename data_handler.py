import numpy as np
from PIL import Image
import os


def load_data(directory, default_label):
    i = 0
    tmp_images = []
    tmp_labels = []
    for f in os.listdir(directory):
        # Opens the test Image
        img = Image.open(directory + f)
        # Converts image to Black and White
        img = img.convert('L')
        # Sets a numpy array of size [width x height] with the greyscale value of each pixel (>200 = White), (<50 = Black))
        arr = np.array(img)
        # arr = np.concatenate(arr).ravel()
        tmp_images.append(np.copy(arr))
        # default_label[i] = 1
        # tmp_labels.append(default_label)
        tmp_labels.append(int(f.split('.')[0]))
        # default_label[i] = 0
        i += 1
        # print('WIDTH: ' + str(arr.shape[1]))
        # print('HEIGHT: ' + str(arr.shape[0]))
        # print(arr)
    return np.array(tmp_images), np.array(tmp_labels)