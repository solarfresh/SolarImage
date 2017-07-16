from tensorflow import Tensor
from tensorflow import exp
from tensorflow import log
from tensorflow import nn
from tensorflow import reduce_sum
from tensorflow import reduce_mean


def genradvers(real, fake, batch_size):
    """
    This is a function generating objective functions in the generative adversarial net.
    We use negative sign for the loss functions because they need to be maximized,
    whereas TensorFlow's optimizer can only do minimization.
    :param real: the probability from actual data passing through the discriminator
    :param fake: the probability from samples from generator and passing through the discriminator
    :param batch_size: the size of running iteration incrementally
    :return:
    """
     # By using softmax gan to check the convergence
    d_target = 1. / batch_size
    g_target = 1. / (batch_size * 2)
    z = reduce_sum(exp(-real)) + reduce_sum(exp(-fake))

    d_loss = reduce_sum(d_target * real) + log(z + 1e-8)
    g_loss = reduce_sum(g_target * real) + reduce_sum(g_target * fake) + log(z + 1e-8)
    # d_loss = -reduce_mean(log(real) + log(1. - fake))
    # g_loss = -reduce_mean(log(fake))
    return d_loss, g_loss


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
