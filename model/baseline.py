#coding=utf-8
from data_preprocess import read_file
import  tensorflow as tf
import numpy as np


root="/home/zsc/下载/data_new/CIKM2017_train/"
# X_train,y_train = read_file.readFile(root+"data_sample.txt")
# print data[0]

x = tf.placeholder('float',[None,15*4*101*101])
y = tf.placeholder('float',[None,1])
lr = tf.placeholder('float')
x = x / 255.
weight = tf.Variable(tf.truncated_normal([15*4*101*101,1],mean=0,stddev=1e-8))
bias = tf.Variable(tf.constant([0.0]))


y_ = tf.matmul(x,weight)+bias

loss_1 = tf.reduce_mean(tf.pow((y-y_),2)) + tf.reduce_sum(tf.square(weight)) * 0.01
loss = tf.pow(loss_1,0.5)

optimizer = tf.train.AdamOptimizer(learning_rate=0.5)

train = optimizer.minimize(loss_1)

init = tf.initialize_all_variables()

num_sample =10000
num_sample_test = 2000
batch_size = 32
epoch = 20
learning_rate=0.001

from data_preprocess.cikm import LineReader
from data_preprocess.read_file import readFile

with tf.Session() as sess:
    sess.run(init)
    for e in xrange(epoch):
        reader = LineReader(root + "train.txt", batch_size=batch_size)
        for i in xrange(num_sample//batch_size):
            try:
                print "========================="
                x_batch,y_batch = reader.next()
                # print y_batch
                x_batch = np.reshape(x_batch,[-1,15*4*101*101])
                y_batch = np.reshape(y_batch,[-1,1])

                [_,loss_,y_hat ,loss_1_] = sess.run([train,loss,y_,loss_1],feed_dict={x:x_batch,y:y_batch,lr:learning_rate})
                print "loss",loss_

            except Exception,e:
                print Exception,":",e

