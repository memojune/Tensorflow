import os
import os.path
import math
import numpy as np
import tensorflow as tf
import cifar10_input

BATCH_SIZE = 128
learning_rate = 0.05
MAX_STEP = 10000

def inference(images):
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,3,96],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.05, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[96],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],
                               strides=[1,2,2,1], padding='SAME',
                               name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0,
                          alpha=0.001/9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 96, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.05, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0,
                          alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1],
                               strides=[1,2,2,1], padding='SAME',
                               name='pooling2')

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.004, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name=scope.name)
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[384, 192],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.004, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[192],
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[192, 10],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.004, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[10],
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.nn.relu(tf.matmul(local4, weights) + biases, name=scope.name)

        return softmax_linear

def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        labels = tf.cast(labels, tf.int64)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                            logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy)

        return loss

def train():
    my_global_step = tf.Variable(0, name='global_step', trainable=False)

    data_dir = 'data/'
    log_dir = 'logs/'

    images, labels = cifar10_input.read_cifar10(data_dir=data_dir,
                                                is_train=True,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)

    logits = inference(images)
    loss = losses(logits, labels)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    saver = tf.train.Saver(tf.global_variables())

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners()

    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            _, loss_value = sess.run([train_op, loss])
            if step % 200 == 0:
                print('Step: %d, loss: %.4f' %(step, loss_value))
            if step % 2000 == 0:
                check_path = os.path.join(log_dir, 'model_ckpt')
                saver.save(sess, check_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

def evaluation():
    with tf.Graph().as_default():
        log_dir = 'logs/'
        test_dir = 'data/'
        n_test = 10000

        images, labels = cifar10_input.read_cifar10(data_dir=test_dir,
                                                    is_train=False,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)

        logits = inference(images)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                num_iter = int(math.ceil(n_test/BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count = np.sum(predictions)
                    step += 1
                    precision = true_count / total_sample_count
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)























































































