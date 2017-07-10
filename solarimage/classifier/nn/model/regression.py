from tensorflow import matmul


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
