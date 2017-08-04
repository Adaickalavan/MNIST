# -*- coding: utf-8 -*-
"""
Jun 2017 
@author: Adaickalavan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def main():
    #Create an interactive session to run tensorflow
    sess = tf.InteractiveSession()
    #Import dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #Create variables and placeholders
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    #Compute layer1
    layer1 = tf.matmul(x, W) + b
#    layer2 = tf.nn.softmax(layer1)
    #Compute cross entropy
#    cross_entropy1 = -tf.reduce_sum(y_ * tf.log(layer2), reduction_indices=[1])
#    cross_entropy2 = tf.reduce_mean(cross_entropy1)
    #Numerically stable implementation of softmax
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=layer1))
    #Implement the training function.
    #Uses gradient descent algorithm with learning rate = 0.5
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
    
    #Initialize tensorflow variables
    tf.global_variables_initializer().run()
    #Let's train -- we'll run the training step 1000 times!
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #We can use tf.equal to check if our prediction matches the truth.
    correct_prediction = tf.equal(tf.argmax(layer1,1), tf.argmax(y_,1))
    #To determine what fraction are correct, we cast to floating point numbers and then take the mean.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #Finally, we ask for our accuracy on our test data.
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    main()