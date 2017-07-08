from tensorflow import Tensor
from tensorflow import nn
from tensorflow import reduce_mean


def cross_entropy(y, y_):
    """
    The raw formulation of cross-entropy
        tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
                                 reduction_indices=[1]))
    can be numerically unstable.
    So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    :param y: the output vector results from nn
    :param y_: the target vector an optimizer tends to achieve
    :return: average loss between target y' and training vector y
    """
    if not isinstance(y, Tensor):
        raise TypeError("The type of y musr be tf.Tensor.")
    if not isinstance(y_, Tensor):
        raise TypeError("The type of y_ musr be tf.Tensor.")
    return reduce_mean(
        nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
