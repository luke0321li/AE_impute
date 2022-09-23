import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, sys
# from tensorflow.examples.tutorials.mnist import input_data
sys.path.append(os.getcwd())
from layer import *

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


class AE:
    def __init__(self, input_dim=10, compress_dim=5, batch_size=20, learning_rate=10e-3, regularization=1e-3, iterations=1000):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.compress_dim = compress_dim
        self.regularization = regularization

        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        self.mask = tf.placeholder(tf.float32, [None, self.input_dim])

        # Mask the input
        self.xm = self.x * self.mask
        # Encoder
        # 2 fully connected layers
        self.w_1, self.b_1 = make_w_and_b([self.input_dim, self.compress_dim])
        self.compressed = tf.nn.relu(tf.matmul(self.xm, self.w_1) + self.b_1)

        # Decoder
        # Another 2 fully connected layers
        self.w_2, self.b_2 = make_w_and_b([self.compress_dim, self.input_dim])
        self.output = tf.matmul(self.compressed, self.w_2) + self.b_2

        # Loss function
        # self.cost = tf.nn.l2_loss(self.xm - self.output * self.mask)
        self.l2 = tf.nn.l2_loss(self.w_1) + tf.nn.l2_loss(self.w_2)
#         self.cost = tf.sqrt(tf.reduce_mean((self.xm - self.output * self.mask) ** 2)) + self.regularization * self.l2
        self.cost = tf.sqrt(tf.reduce_mean((self.x - self.output) ** 2)) + self.regularization * self.l2
        self.objective = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def train(self, data, masks, test, test_masks):
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        data_size = len(data)
        costs = np.zeros(self.iterations)
        with tf.Session() as sess:
            sess.run(init)
            current_index = 0
            for i in range(self.iterations):
                batch = data.take(np.arange(current_index, current_index + self.batch_size), axis=0, mode='wrap')
                mask = masks.take(np.arange(current_index, current_index + self.batch_size), axis=0, mode='wrap')
                current_index += self.batch_size
                sess.run(self.objective, feed_dict={self.x: batch, self.mask: mask})
                cost = self.cost.eval(feed_dict={self.x: batch, self.mask: mask})
                costs[i] = cost
            path = saver.save(sess, "./model_1")
            pred = self.output.eval(feed_dict={self.x: test, self.mask: test_masks})
        return costs, pred, test
    
    def test(self, data, masks):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "./model_1")
            pred = self.output.eval(feed_dict={self.x: data, self.mask: masks})
        return pred



if __name__ == '__main__':
    # data = np.loadtxt("/Users/runjiali/documents/genomics/projects/impute/impute_proj/E001_H3K27me3.marks.train")
    # masks = np.loadtxt("/Users/runjiali/documents/genomics/projects/impute/impute_proj/E001_H3K27me3.marks.masks")
    data = np.zeros((1000, 3))
    data[:, 0] = np.random.randn(1000)
    data[:, 1] = data[:, 0] * 2 + np.random.normal(0, 0.05, data.shape[0])
    data[:, 2] = 2 * (data[:, 1] - data[:, 0]) + np.random.normal(0, 0.05, data.shape[0])
    masks = np.random.binomial(1, 0.8, data.shape)
    # masks = np.ones(data.shape)
    AE = AE(input_dim = data.shape[1], intermediate_dim=50, compress_dim=50, batch_size=900, learning_rate=0.001, iterations=1000)
    c = AE.train(data[:990], masks[:990], data[990:], masks[990:])
    print(c[-10:])
    plt.plot(c)
    plt.show()
