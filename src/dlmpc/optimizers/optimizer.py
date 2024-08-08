import tensorflow as tf

def optimizer(optimizer_name, learning_rate, du_bound=None,exponential_decay=False):
    if exponential_decay:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=1, decay_rate=.99)
    if optimizer_name == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipvalue = du_bound)
    elif optimizer_name == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue = du_bound)
    elif optimizer_name == 'adagrad':
        opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, clipvalue = du_bound)
    elif optimizer_name == 'adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate, clipvalue=du_bound)
    elif optimizer_name == 'adamax':
        opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate, clipvalue = du_bound)
    elif optimizer_name == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, clipvalue = du_bound)
    return opt



if __name__ == '__main__':
    # optimizer(optimizer_name='adam', learning_rate=0.1, du_bound=0.1, exponential_decay=True)
    optimizer(optimizer_name='sgd', learning_rate=0.1, du_bound=0.1, exponential_decay=True)

