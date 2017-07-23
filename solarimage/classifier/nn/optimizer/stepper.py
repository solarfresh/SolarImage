from tensorflow import train


def adam_optimizer(learning_rate, objective, var_list):
    """
    A stochastic gradient decent method
    :param objective:  an objective function
    :param weights:  weights used in the model
    :param var_list: the list of training variables
    :return:
    """
    return train.AdamOptimizer(learning_rate=learning_rate)\
        .minimize(objective, var_list=var_list)


def gradient_decent(learning_rate, objective, var_list):
    """
     This is a first-order iterative optimization algorithm for finding the minimum of a function.
     To find a local minimum of a function using gradient descent, one takes steps proportional to
     the negative of the gradient (or of the approximate gradient) of the function at the current point.
    :param learning_rate: A Tensor or a floating point value. The learning rate to use.
    :param objective: An objective function used to be minimized
    :param var_list: the list of training variables
    :return: An Operation that updates the variables
    """
    return train.GradientDescentOptimizer(learning_rate)\
        .minimize(objective, var_list=var_list)
