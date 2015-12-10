import pandas as pd
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
import theano


def read_bezier_dataset(filenames):
    X = map(read_from_bezier_file, filenames)
    X = map(preprocessed_bezier, X)
    return X


def read_images(filenames, img_folder="png", ext="png"):
    names = map(raw_name, filenames)
    names = [os.path.join(img_folder, name) + "." + ext
             for name in names]
    return map(imread, names)


def resize_images(images, size):
    return [resize(image, size).tolist() for image in images]


def raw_name(filename):
    return os.path.basename(filename.split(".")[0])


def read_from_bezier_file(filename):
    names = ["p1x", "p1y", "p2x", "p2y", "p3x", "p3y", "p4x", "p4y"]
    df = pd.read_table(filename,
                       skiprows=1,
                       sep=' ',
                       names=names)
    return df.values


def relative_position(bezier_values):
    z = np.zeros((1, 8), dtype=theano.config.floatX)
    bezier_values_ = np.concatenate((z, bezier_values), axis=0)
    return bezier_values_[1:] - bezier_values_[0:-1]


def absolute_position(bezier_values):
    return bezier_values.cumsum(axis=0)


def preprocessed_bezier(o):
    s = slice(0, len(o))
    m1 = (0, 2, 4, 6)
    m2 = (1, 3, 5, 7)
    o = o.astype(np.float32)
    for m in (m1, m2):
        o[s, m] = o[s, m] - o[s, m].min()
        o[s, m] = o[s, m] / o[s, m].max()
    o = np.concatenate((np.zeros((1, 8)), o), axis=0)
    o = np.concatenate((o, np.zeros((1, 8))), axis=0)
    o = o.astype(np.float32)
    return o

if __name__ == "__main__":
    import glob
    filenames = glob.glob("svg/*.txt")
    x = read_bezier_dataset(filenames)
    img = read_images(filenames)
