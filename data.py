import pandas as pd
import numpy as np


def read_bezier_dataset(filenames):
    X = []
    for filename in filenames:
        x = read_from_bezier_file(filename)
        x = preprocessed_bezier(x)
        X.append(x)
    return X


def read_from_bezier_file(filename):
    names = ["p1x", "p1y", "p2x", "p2y", "p3x", "p3y", "p4x", "p4y"]
    df = pd.read_table(filename,
                       skiprows=1,
                       sep=' ',
                       names=names)
    return df.values


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
    print(x[1])
