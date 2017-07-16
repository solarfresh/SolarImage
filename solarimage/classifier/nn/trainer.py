import numpy as np
from solarimage.initializer.random import xavier_init
from .model import regression
from .optimizer import objective
from .optimizer import stepper
from numpy import arange
from numpy import concatenate
from numpy import float32
from numpy import multiply
from numpy import random
from tensorflow import float32
from tensorflow import global_variables_initializer
from tensorflow import placeholder
from tensorflow import Session
from tensorflow import Variable
from tensorflow import zeros
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 dtype=dtypes.float32,
                 reshape=False,
                 seed=None):
        """Construct a DataSet.
                `dtype` can be either `uint8` to leave the input as `[0, 255]`,
                 or `float32` to rescale into `[0, 1]`.
                """
        seed1, seed2 = random_seed.get_seed(seed)
        random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)

        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(float32)
            images = multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = arange(self._num_examples)
            random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return concatenate((images_rest_part, images_new_part), axis=0), concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


class TrainVar(DataSet):
    """
    A class defining training variables
    """
    def __init__(self, sample_size=100, hidden_size=128, *args, **kwargs):
        DataSet.__init__(self, *args, **kwargs)
        self.image_size = self.images.shape[1]
        self.labels_size = self.labels.shape[1]
        self.sample_size = sample_size
        self.hidden_size = hidden_size
        x = placeholder(float32, shape=[None, self.image_size], name='src_var')
        self.train_var = x

    def get_genradvers_var(self):
        if not self.sample_size:
            raise ValueError("The size of sample is necessary.")
        # Generator Net
        z = placeholder(float32, shape=[None, self.sample_size], name='sample_var')
        g_w1 = Variable(xavier_init([self.sample_size, self.hidden_size]), name='gen_weight_1')
        g_b1 = Variable(zeros(shape=[self.hidden_size]), name='gen_bias_1')
        g_w2 = Variable(xavier_init([self.hidden_size, self.image_size]), name='gen_weight_2')
        g_b2 = Variable(zeros(shape=[self.image_size]), name='gen_bias_2')
        theta_g = [g_w1, g_b1, g_w2, g_b2]
        # Discriminator Net
        x = placeholder(float32, shape=[None, self.image_size], name='src_var')
        d_w1 = Variable(xavier_init([self.image_size, self.hidden_size]), name='dis_weight_1')
        d_b1 = Variable(zeros(shape=[self.hidden_size]), name='dis_bias_1')
        d_w2 = Variable(xavier_init([self.hidden_size, 1]), name='dis_weight_2')
        d_b2 = Variable(zeros(shape=[1]), name='dis_bias_2')
        theta_d = [d_w1, d_b1, d_w2, d_b2]
        self.train_var = z, theta_g, x, theta_d
        return self.train_var


class TrainModel(TrainVar):
    """
    A class defining training models
    """
    def __init__(self, *args, **kwargs):
        TrainVar.__init__(self, *args, **kwargs)

    def get_genradvers_loss(self, batch_size):
        z, theta_g, x, theta_d = self.train_var
        sample, _ = regression.log_linear(z, *theta_g)
        real_prob, real_log_prob = regression.log_linear(x, *theta_d)
        fake_prob, fake_log_prob = regression.log_linear(sample, *theta_d)
        return objective.genradvers(real_log_prob, fake_log_prob, batch_size)


class TrainOpt(TrainModel):
    def __init__(self, learning_rate=0.5, *args, **kwargs):
        TrainModel.__init__(self, *args, **kwargs)
        self.learning_rate = learning_rate

    def get_genradvers_solver(self, d_loss, theta_d, g_loss, theta_g):
        # Only update D(X)'s parameters, so var_list = theta_D
        d_solver = stepper.adam_optimizer(self.learning_rate, d_loss, theta_d)
        # Only update G(X)'s parameters, so var_list = theta_G
        g_solver = stepper.adam_optimizer(self.learning_rate, g_loss, theta_g)
        return d_solver, g_solver


class TrainIter(TrainOpt):
    def __init__(self, batch_size, iter_max, *args, **kwargs):
        TrainOpt.__init__(self, *args, **kwargs)
        self.batch_size = batch_size
        self.iter_max = iter_max

    def run_genradvers(self):
        def sample_z(m, n):
            # Uniform prior for G(Z)'
            return np.random.uniform(-1., 1., size=[m, n])

        z, theta_g, x, theta_d = self.get_genradvers_var()
        d_loss, g_loss = self.get_genradvers_loss(self.batch_size)
        d_solver, g_solver = self.get_genradvers_solver(d_loss, theta_d, g_loss, theta_g)

        d_loss_list = []
        g_loss_list = []
        with Session() as sess:
            sess.run(global_variables_initializer())
            for _ in range(self.iter_max):
                #  train process
                batch_xs, batch_ys = self.next_batch(self.batch_size, shuffle=False)
                zs = sample_z(self.batch_size, self.sample_size)
                _, d_loss_curr = sess.run([d_solver, d_loss], feed_dict={x: batch_xs, z: zs})
                _, g_loss_curr = sess.run([g_solver, g_loss], feed_dict={x: batch_xs, z: zs})
                d_loss_list.append(d_loss_curr)
                g_loss_list.append(g_loss_curr)
        return d_loss_list, g_loss_list


#  Will need  a configure file in the future
MODEL_DICT = {
    "llr": regression.log_linear,
    "lr": regression.linear
}
OPTIMIZE_DICT = {
    "gd": stepper.gradient_decent,
    "adam": stepper.adam_optimizer
}


def runner(train_dataset, batch_size, model='lr', optimizer='gd',
           iter_max=1000, learning_rate=0.5, shuffle=True):
    """
    An runner of optimizer to approach the minimum error
    :param train_dataset: the data set used to learn
    :param batch_size: the size of batch obtained from dataset
    :param model: a model used to build the relation between input and output
    :param optimizer: an optimization to obtain the coefficients of neuron
    :param iter_max: the max times of iteration
    :param learning_rate: the size of step of gradient decent
    :param shuffle: to randomize the data set initially
    :return: a list of accuracy
    """
    #  Configuration
    images_size = train_dataset.images.shape[1]
    labels_size = train_dataset.labels.shape[1]
    #  Allocate images column vector
    x = placeholder(float32, [None, images_size])
    w = Variable(zeros([images_size, labels_size]), name="weights")
    b = Variable(zeros([labels_size]), name="bias")
    # Create a model maps labels from images
    y = MODEL_DICT[model](x, w, b)
    #  Allocate labels column vector
    y_ = placeholder(float32, [None, labels_size])
    #  Define the loss function
    cost = objective.cross_entropy(y, y_)
    #  Create an optimization stepper
    train_step = OPTIMIZE_DICT[optimizer](learning_rate, cost)

    loss_list = []
    with Session() as sess:
        sess.run(global_variables_initializer())
        for _ in range(iter_max):
            #  train process
            batch_xs, batch_ys = train_dataset.next_batch(batch_size, shuffle=shuffle)
            _, weights, bias, loss_val = sess.run([train_step, w, b, cost], feed_dict={x: batch_xs, y_: batch_ys})
            loss_list.append(loss_val)
    return loss_list, weights, bias
