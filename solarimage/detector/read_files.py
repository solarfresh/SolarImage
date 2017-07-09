import cv2
import numpy as np
import os
import pandas as pd
from os import path


def get_images_labels(file_path, gray=True):
    files = pd.Series(os.listdir(file_path))
    classes = files.str.split('[._]').apply(lambda x: x[0]).unique()
    basis = pd.DataFrame(data=np.eye(classes.size), columns=classes)
    if gray:
        images = files\
            .apply(lambda x: cv2.imread(path.join(file_path, x), 0))\
            .apply(lambda x: cv2.resize(x, (125, 125)).ravel()) #     for simple, we force all in the same size
    else:
        images = files.apply(lambda x: cv2.imread(path.join(file_path, x)))
    labels = files.str.split('[._]').apply(lambda x: basis[x[0]].values)
    return np.vstack(images), np.vstack(labels)
