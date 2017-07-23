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
from tensorflow import Session
from tensorflow import Variable
from tensorflow import constant
from tensorflow import float32
from tensorflow import global_variables_initializer
from tensorflow import layers
from tensorflow import nn
from tensorflow import placeholder
from tensorflow import reshape
from tensorflow import truncated_normal
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
    def __init__(self, rank=1, *args, **kwargs):
        DataSet.__init__(self, *args, **kwargs)
        self.rank = rank
        self.labels_size = self.labels.shape[1]
        self.images_size = self.images.shape[1]
        if rank > 1:
            self.images_width = self.images.shape[0]
            self.images_height = self.images.shape[1]
        if rank > 2:
            self.images_depth = self.images.shape[2]

    def get_convolution_var(self):
        if self.rank < 2:
            raise ValueError("The shape of images must be 2-D")
        x = placeholder(float32, shape=[self.images_width, self.images_height], name='src_var')
        # patch 5x5, in size 1, and filter size 32
        conv_w1 = Variable(truncated_normal([5, 5, 1, 32], stddev=0.1), name='conv_weight_1')
        conv_b1 = Variable(constant(0.1, shape=[32]), name='conv_bias_1')
        # patch 5x5, in size 32, and filter size 64
        conv_w2 = Variable(truncated_normal([5, 5, 32, 64], stddev=0.1), name='conv_weight_2')
        conv_b2 = Variable(constant(0.1, shape=[64]), name='conv_bias_2')
        theta_conv = [conv_w1, conv_b1, conv_w2, conv_b2]
        return x, theta_conv

    def get_genradvers_var(self, sample_size=100, hidden_size=128):
        # Generator Net
        z = placeholder(float32, shape=[None, sample_size], name='sample_var')
        g_w1 = Variable(xavier_init([sample_size, hidden_size]), name='gen_weight_1')
        g_b1 = Variable(zeros(shape=[hidden_size]), name='gen_bias_1')
        g_w2 = Variable(xavier_init([hidden_size, self.images_size]), name='gen_weight_2')
        g_b2 = Variable(zeros(shape=[self.images_size]), name='gen_bias_2')
        theta_g = [g_w1, g_b1, g_w2, g_b2]
        # Discriminator Net
        x = placeholder(float32, shape=[None, self.images_size], name='src_var')
        d_w1 = Variable(xavier_init([self.images_size, hidden_size]), name='dis_weight_1')
        d_b1 = Variable(zeros(shape=[hidden_size]), name='dis_bias_1')
        d_w2 = Variable(xavier_init([hidden_size, 1]), name='dis_weight_2')
        d_b2 = Variable(zeros(shape=[1]), name='dis_bias_2')
        theta_d = [d_w1, d_b1, d_w2, d_b2]
        return z, theta_g, x, theta_d

    def get_simple_var(self):
        x = placeholder(float32, shape=[None, self.images_size], name='src_var')
        w = Variable(zeros([self.images_size, self.labels_size]), name="weights")
        b = Variable(zeros([self.labels_size]), name="bias")
        return x, w, b


class TrainModel(TrainVar):
    """
    A class defining training models
    """
    def __init__(self, *args, **kwargs):
        TrainVar.__init__(self, *args, **kwargs)

    def convolutional_nets(self, batch_size, images, labels, mode):
        def flatten_layer(layer):
            layer_shape = layer.get_shape()
            num_features = layer_shape[1:4].num_elements()

            layer_flat = reshape(layer, [-1, num_features])
            return layer_flat, num_features

        # Convolutional Layer #1
        #  strides:  [batch_stride x_stride y_stride depth_stride]
        conv1 = layers.conv2d(
            inputs=images,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=nn.relu)

        # Pooling Layer #1
        #  If some cases, people slide the windows by more than 1 pixel. This number is called stride.
        pool1 = layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=nn.relu)
        pool2 = layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        # pool2_flat = reshape(pool2, [-1, 7 * 7 * 64])
        pool2_flat, _ = flatten_layer(pool2)
        dense = layers.dense(inputs=pool2_flat, units=1024, activation=nn.relu)
        dropout = layers.dropout(
            inputs=dense, rate=0.4, training=mode)

        # Logits Layer
        logits = layers.dense(inputs=dropout, units=10)
        return True

    def get_genradvers_loss(self, batch_size, x, sample, theta_d):
        real_prob, real_log_prob = regression.log_linear(x, *theta_d)
        fake_prob, fake_log_prob = regression.log_linear(sample, *theta_d)
        return objective.genradvers(real_log_prob, fake_log_prob, batch_size)

    def linear_regression(self, x, w, b):
        y = regression.linear(x, w, b)
        y_ = placeholder(float32, [None, self.labels_size])
        return y, y_


class TrainOpt(TrainModel):
    def __init__(self, *args, **kwargs):
        TrainModel.__init__(self, *args, **kwargs)
        self.solver = {
            "gd": stepper.gradient_decent,
            "sgd": stepper.adam_optimizer,
        }

    def get_genradvers_solver(self, learning_rate, d_loss, theta_d, g_loss, theta_g):
        # Only update D(X)'s parameters, so var_list = theta_D
        d_solver = stepper.adam_optimizer(learning_rate, d_loss, theta_d)
        # Only update G(X)'s parameters, so var_list = theta_G
        g_solver = stepper.adam_optimizer(learning_rate, g_loss, theta_g)
        return d_solver, g_solver

    def get_simple_solver(self, learning_rate, loss, var_list, method="gd"):
        return self.solver[method](learning_rate, loss, var_list)


class TrainIter(TrainOpt):
    def __init__(self, *args, **kwargs):
        TrainOpt.__init__(self, *args, **kwargs)

    def run_genradvers(self,
                       batch_size=100,
                       iter_max=1e4,
                       learning_rate=1e-3,
                       sample_size=100,
                       hidden_size=128):
        def sample_z(m, n):
            # Uniform prior for G(Z)'
            return np.random.uniform(-1., 1., size=[m, n])

        z, theta_g, x, theta_d = self.get_genradvers_var(sample_size=sample_size,
                                                         hidden_size=hidden_size)
        sample, _ = regression.log_linear(z, *theta_g)
        d_loss, g_loss = self.get_genradvers_loss(batch_size, x, sample, theta_d)
        d_solver, g_solver = self.get_genradvers_solver(learning_rate, d_loss, theta_d, g_loss, theta_g)

        d_loss_list = []
        g_loss_list = []
        with Session() as sess:
            sess.run(global_variables_initializer())
            for _ in range(int(iter_max)):
                #  train process
                batch_xs, batch_ys = self.next_batch(batch_size, shuffle=False)
                zs = sample_z(batch_size, sample_size)
                _, d_loss_curr = sess.run([d_solver, d_loss], feed_dict={x: batch_xs, z: zs})
                _, g_loss_curr = sess.run([g_solver, g_loss], feed_dict={x: batch_xs, z: zs})
                d_loss_list.append(d_loss_curr)
                g_loss_list.append(g_loss_curr)
        return d_loss_list, g_loss_list

    def run_simple_nn(self,
                      batch_size=100,
                      iter_max=1e4,
                      learning_rate=1e-3,
                      solver_method='gd'):
        x, w, b = self.get_simple_var()
        y, y_ = self.linear_regression(x, w, b)
        loss = objective.cross_entropy(y, y_)
        solver = self.get_simple_solver(learning_rate, loss, [w, b], method=solver_method)
        loss_list = []
        with Session() as sess:
            sess.run(global_variables_initializer())
            for _ in range(int(iter_max)):
                #  train process
                batch_xs, batch_ys = self.next_batch(batch_size, shuffle=True)
                _, loss_val, weights, bias = sess.run([solver, loss, w, b],
                                                      feed_dict={x: batch_xs, y_: batch_ys})
                loss_list.append(loss_val)
        return loss_list, weights, bias
