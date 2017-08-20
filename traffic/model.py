import tensorflow as tf
from tensorflow.contrib.layers import flatten


rate = 0.001
tf.set_random_seed(202)
n_classes= 43
summ_dir = '/tmp/tensorflow/'


def conv2d_relu(x, w_shape, strides=1, padding='VALID'):
    W = tf.Variable(tf.truncated_normal(shape=w_shape, mean=mu, stddev=sigma))
    b = tf.Variable(tf.zeros(w_shape[-1]))
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
                     padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def fc_layer(x, num_outputs):
    fc1 = tf.contrib.layers.fully_connected(x, num_outputs)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    return fc1

def LeNet(x, n_channels=3, n_classes=43):

    conv1 = conv2d_relu(x, (5, 5, n_channels, 6))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2 = conv2d_relu(conv1, (5, 5, 6, 16))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2 = tf.contrib.layers.batch_norm(conv2)

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)
    fc1 = fc_layer(fc0, 120)
    fc2 = fc_layer(fc1, 84)
    logits = tf.contrib.layers.fully_connected(fc2, n_classes)

    return logits


def create_variables():
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)

    with tf.name_scope('metrics'):
        logits = LeNet(x)
        tf.summary.histogram('logits', logits)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y,
                                                                logits=logits)

        loss_operation = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss_operation', loss_operation)
        optimizer = tf.train.AdamOptimizer(learning_rate=rate)
        training_operation = optimizer.minimize(loss_operation)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))

        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), )
        tf.summary.scalar('accuracy', accuracy_operation)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summ_dir + '/train',
                                          sess.graph)
    test_writer = tf.summary.FileWriter(summ_dir + '/test')


class DataHolder():

    def __init__(self, X_train, X_valid, y_train, y_valid):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid


def train_model(d, save_name='lenet_final', epochs=EPOCHS, batch_size=BATCH_SIZE):
    create_variables()
    train_accs = {}
    valid_accs = {}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        print("Training...")
        for i in range(epochs):
            X, y = shuffle(d.X_train, d.y_train)
            for j, offset in enumerate(range(0, num_examples, batch_size)):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X[offset:end], y[offset:end]
                sess.run(training_operation,
                         feed_dict={x: batch_x, y: batch_y, keep_prob: .5})

            print("EPOCH {} ...".format(i+1))
            a, b = log_accuracy(d, verbose=True)
            train_accs[i] = a
            valid_accs[i] = b

        saver = tf.train.saver()
        saver.save(sess, save_name)
        print("Model saved")
    return train_accs, val_accs