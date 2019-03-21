import numpy as np
from PIL import Image
import os


def load_data(directory, default_label):
    tmp_images = []
    tmp_labels = []
    for f in os.listdir(directory):
        # Checks if the file fits input standards (#.bmp files)
        if f[0].isdigit() and os.path.splitext(f)[1] == '.bmp':
            # Opens the test Image
            img = Image.open(directory + f)
            # Converts image to Black and White
            img = img.convert('L')
            # Sets a numpy array of size [width x height] with the greyscale value of each pixel
            # (>200 = White), (<50 = Black))
            arr = np.array(img)
            # Turns the 2d array into a 1d array (this allows each element as input to the neural network)
            arr = np.concatenate(arr).ravel()

            # takes the first character of the file, classifies it as that (eg. 0_test.bmp would classify as 0)
            default_label[int(f[0])] = 1
            tmp_images.append(np.copy(arr))
            tmp_labels += [np.copy(default_label)]
            default_label[int(f[0])] = 0
        else:
            print("please label input files with the digit label at the front")

    return np.array(tmp_images), np.array(tmp_labels)