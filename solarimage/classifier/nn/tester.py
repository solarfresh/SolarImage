from .model import regression
from tensorflow import argmax
from tensorflow import cast
from tensorflow import equal
from tensorflow import float32
from tensorflow import global_variables_initializer
from tensorflow import reduce_mean
from tensorflow import Session
from tensorflow import placeholder


MODEL_DICT = {
    "lr": regression.linear
}


def runner(test_dataset, weights, bias, model='lr'):
    """
    An runner to exam results
    :param test_dataset: the data set used to test
    :param weights: obtaine after training
    :param bias: obtaine after training
    :param model: a model trained
    :return:  accuracy
    """
    #  Configuration
    images_size = test_dataset.images.shape[1]
    labels_size = test_dataset.labels.shape[1]
    #  Allocate images column vector
    x = placeholder(float32, [None, images_size])
    # Create a model maps labels from images
    y = MODEL_DICT[model](x, weights, bias)
    #  Allocate labels column vector
    y_ = placeholder(float32, [None, labels_size])

    with Session() as sess:
        # sess.run(tf.global_variables_initializer())
        correct_prediction = equal(argmax(y, 1), argmax(y_, 1))
        accuracy = reduce_mean(cast(correct_prediction, float32))
        return sess.run(accuracy, feed_dict={x: test_dataset.images,
                                            y_: test_dataset.labels})