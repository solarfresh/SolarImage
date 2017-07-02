from tensorflow import train


def gradient_decent(learning_rate, objective):
    """
     This is a first-order iterative optimization algorithm for finding the minimum of a function.
     To find a local minimum of a function using gradient descent, one takes steps proportional to
     the negative of the gradient (or of the approximate gradient) of the function at the current point.
    :param learning_rate: A Tensor or a floating point value. The learning rate to use.
    :param objective: An objective function used to be minimized
    :return: An Operation that updates the variables
    """
    return train.GradientDescentOptimizer(learning_rate).minimize(objective)


def runner(x, y_, train_stepper, cost_fn, iter_max, batch_size,
           train_dataset, session, shuffle=True):
    """
    An runner of optimizer to approach the minimum error
    :param x: the column vector of images in pixel
    :param y_: target colimn vector from batch
    :param train_stepper: the algorithm of optimization
    :param the cost function used to approach the extrema values
    :param iter_max: the max times of iteration
    :param batch_size: the size of batch obtained from dataset
    :param train_dataset: the dataset used to be optimized
    :param session: a session allocated from tensorflow
    :return: a list of accuracy
    """
    monitor = []
    for _ in range(iter_max):
        #  train process
        batch_xs, batch_ys = train_dataset.next_batch(batch_size, shuffle=shuffle)
        _, loss_val = session.run([train_stepper, cost_fn], feed_dict={x: batch_xs, y_: batch_ys})
        monitor.append(loss_val)
    return monitor
