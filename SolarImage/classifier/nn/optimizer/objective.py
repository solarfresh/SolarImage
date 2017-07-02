from tensorflow import Tensor
from tensorflow import float32
from tensorflow import nn
from tensorflow import placeholder
from tensorflow import reduce_mean


def cross_entropy(y):
    """
    The raw formulation of cross-entropy
        tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
                                 reduction_indices=[1]))
    can be numerically unstable.
    So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    :param y: the output vector results from nn
    :return: average loss between target y' and training vector y
    """
    if not isinstance(y, Tensor):
        raise TypeError("The type of input musr be tf.Tensor.")
    y_ = placeholder(float32, y.shape)
    return reduce_mean(
      nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))