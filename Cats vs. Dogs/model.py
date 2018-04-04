import tensorflow as tf

def inference(images, batch_size, n_classes):

    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape = [3, 3, 3, 16], # 卷积核长、宽、通道入数、通道出数
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
                                           # strides: batch, width, height, channel
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('pooling_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],
                               strides=[1,2,2,1], padding='SAME',
                               name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0,
                          alpha=0.001/9, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0,
                          alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME',
                               name='pooling2')

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('soft_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.nn.relu(tf.matmul(local4, weights) + biases, name='softmax_linear')

    return softmax_linear

def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
    return loss

def training(loss, learningrate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learningrate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1) # 返回的是logits中最大值的位置与labels比较
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
    return accuracy