import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# data
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise                          # shape (100, 1) + some noise

xs = tf.cast(x, tf.float32)
ys = tf.cast(y, tf.float32)

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout  
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs


# define placeholder for inputs to network  
keep_prob = tf.placeholder(tf.float32)


l1 = add_layer(xs, 1, 20, activation_function=tf.nn.relu)
prediction = add_layer(l1, 20, 1)

loss = tf.losses.mean_squared_error(predictions=prediction, labels=ys)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

plt.ion()

for i in range(500):
    # here to determine the keeping probability  
    _, res = sess.run([train_step, prediction], feed_dict={keep_prob: 1})
    if i % 10 == 0:
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, res)
        plt.pause(0.1)

plt.ioff()
plt.show()