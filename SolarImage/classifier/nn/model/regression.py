from tensorflow import Variable
from tensorflow import float32
from tensorflow import matmul
from tensorflow import placeholder
from tensorflow import zeros


def linear(D, K):
    """
    The image x has all of its pixels flattened out to a single column vector of shape [D x 1].
    The matrix W (of size [K x D]), and the vector b (of size [K x 1]) are the parameters of the function.
    :param D:  size of image x in a single column vector
    :param K:  size of label to be classified
    :return:
    """
    x = placeholder(float32, [None, D])
    W = Variable(zeros([D, K]))
    b = Variable(zeros([K]))
    return matmul(x, W) + b