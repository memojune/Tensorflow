import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

t_start = time.time()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def compute_accuracy(v_xs, v_ys):
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1})# 每一行是每个样本的预测结果
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))# argmax得到的是位置，1为每行最大值所在列
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    return result

def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 784])# 28*28
ys = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(xs, [-1,28,28,1])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

# 可以考虑使用tf.nn.softmax_cross_entropy_with_logits()
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:1})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))

t_end = time.time()
print(t_end - t_start)
