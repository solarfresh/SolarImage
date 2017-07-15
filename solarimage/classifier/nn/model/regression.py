from tensorflow import matmul
from tensorflow import nn


def genradvers(x, w1, b1, w2, b2):
    """
    Here, we collect models of generative adversarial nets
    :return:
    """
    h1 = nn.relu(matmul(x, w1) + b1)
    #  Be not sure the movitation of using the second regression
    #  It is necessary to provide an reason later.
    log_prob = matmul(h1, w2) + b2
    return nn.sigmoid(log_prob)


def linear(x, w, b):
    """
    The image x has all of its pixels flattened out to a single column vector of shape [d x 1].
    The matrix W (of size [k x d]), and the vector b (of size [k x 1]) are the parameters of the function.
    :param x:  placeholder of a column vector as an input with the dimension D
    :param w:  weight
    :param b:  bias
    :return:
    """
    return matmul(x, w) + b
