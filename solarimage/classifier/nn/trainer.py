from .model import regression
from .optimizer import objective
from .optimizer import stepper
from tensorflow import float32
from tensorflow import global_variables_initializer
from tensorflow import InteractiveSession
from tensorflow import placeholder


MODEL_DICT = {
    "lr": regression.linear
}
OPTIMIZE_DICT = {
    "gd": stepper.gradient_decent
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
    # Create a model maps labels from images
    y = MODEL_DICT[model](x, images_size, labels_size)
    #  Allocate labels column vector
    y_ = placeholder(float32, [None, labels_size])
    #  Define the loss function
    cost = objective.cross_entropy(y, y_)
    #  Create an optimization stepper
    train_step = OPTIMIZE_DICT[optimizer](learning_rate, cost)
    #  Initialize a session
    session = InteractiveSession()

    loss_list = []
    global_variables_initializer().run()
    for _ in range(iter_max):
        #  train process
        batch_xs, batch_ys = train_dataset.next_batch(batch_size, shuffle=shuffle)
        _, loss_val = session.run([train_step, cost], feed_dict={x: batch_xs, y_: batch_ys})
        loss_list.append(loss_val)
    return loss_list
