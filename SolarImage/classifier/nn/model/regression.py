from tensorflow import Variable
from tensorflow import matmul
from tensorflow import zeros


def linear(x, d, k):
    """
    The image x has all of its pixels flattened out to a single column vector of shape [d x 1].
    The matrix W (of size [k x d]), and the vector b (of size [k x 1]) are the parameters of the function.
    :param x:  placeholder of a column vector as an input with the dimension D
    :param d:  size of image x in a single column vector
    :param k:  size of label to be classified
    :return:
    """
    w = Variable(zeros([d, k]))
    b = Variable(zeros([k]))
    return matmul(x, w) + b
