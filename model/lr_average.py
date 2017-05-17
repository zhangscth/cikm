#coding=utf-8
from data_preprocess import read_file
import  tensorflow as tf
import numpy as np
np.random.seed(1024)


root="/home/zsc/下载/data_new/CIKM2017_train/"
# X_train,y_train = read_file.readFile(root+"data_sample.txt")
# print data[0]

x = tf.placeholder('float',[None,15*4*202])
y = tf.placeholder('float',[None,1])
lr = tf.placeholder('float')
x_scale = x / 255.
weight = tf.Variable(tf.truncated_normal([15*4*202,1],mean=1e-8,stddev=1e-6))
bias = tf.Variable(tf.constant([0.01]))


y_ = tf.matmul(x_scale,weight)+bias

loss_1 = tf.reduce_mean(tf.pow((y-y_),2)) + tf.reduce_sum(tf.square(weight)) * 0.01
loss = tf.pow(loss_1,0.5)

optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01)

train = optimizer.minimize(loss_1)

init = tf.initialize_all_variables()

num_sample =10000
num_sample_test = 2000
batch_size = 64
epoch = 50
learning_rate=0.001

from data_preprocess.cikm import LineReader_average
from data_preprocess.read_file import readFile

with tf.Session() as sess:
    sess.run(init)
    for e in xrange(epoch):
        reader = LineReader_average(root + "train.txt", batch_size=batch_size)
        for i in xrange(num_sample//batch_size):
            print i
            try:
                print "========================="
                x_batch,y_batch = reader.next()
                # print y_batch
                x_batch = np.reshape(x_batch,[-1,15*4*202])
                y_batch = np.reshape(y_batch,[-1,1])

                [_,loss_,y_hat ,loss_1_] = sess.run([train,loss,y_,loss_1],feed_dict={x:x_batch,y:y_batch,lr:learning_rate})
                print "loss",loss_

            except Exception,e:
                print Exception,":",e



    result = []
    with open("/home/zsc/下载/data_new (3)/CIKM2017_testA/testA.txt", 'r') as f:
        i = 0
        while (1):
            line = f.readline()
            line = line.strip()

            if line == "" or line is None:
                break
            lis = line.split(",")
            i += 1
            if i % 100 == 0:
                print i
            # preprocess

            X_train = map(float, lis[2].split(' '))
            x_train = np.reshape(X_train, [-1, 60, 101, 101])
            # print x_train_feature.flatten()
            # x_train_feature = np.reshape(x_train, [-1, 15*4 * 101* 101])
            x_train_mean1 = np.mean(x_train, axis=2).reshape([1, -1])  # 60 *101
            x_train_mean2 = np.mean(x_train, axis=3).reshape([1, -1])  # 60 * 101

            x_train = np.concatenate([x_train_mean1, x_train_mean2], axis=1)  # 60 * 202维
            predict = sess.run(y_, feed_dict={x: x_train})
            # print predict
            result.append(predict[0][0])
            # data.append(line)
    # data = np.array(data, dtype='float32')

    f = open("result.csv", 'wb')
    for line in result:
        if line < 0:
            line = 0
        f.write(str(line) + '\n')
    f.close()
