from tensorflow import random_normal
from tensorflow import sqrt


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / sqrt(in_dim / 2.)
    return random_normal(shape=size, stddev=xavier_stddev)