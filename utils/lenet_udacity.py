import tensorflow as tf
from tensorflow.contrib.layers import flatten

keep_prob = tf.placeholder(tf.float32)
mu = 0
sigma = 0.1
DATA_URL = 'https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip'

def get_n_params():
    total_parameters = 0
    trainaible_vars = tf.trainable_variables()
    for variable in trainaible_vars:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()

        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return {'n_params': total_parameters, 'n_vars': len(trainaible_vars)}


def evaluate(X_data, y_data, num_examples=None):
    if num_examples is None:
        num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation,
                            feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: 1.})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def log_accuracy(verbose=False):
    validation_accuracy = evaluate(X_valid, y_valid)
    train_accuracy = evaluate(X_train, y_train)
    if verbose:
        print("Train={:.3f}, Validation={:.3f}".format(
            train_accuracy, validation_accuracy))
    return train_accuracy, validation_accuracy