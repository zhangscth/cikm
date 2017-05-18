#coding=utf-8
from data_preprocess import read_file
import  tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pdb


root="/home/zsc/下载/data_new/CIKM2017_train/"
X_train,y_train = read_file.readFile(root+"data_sample.txt")
# print data[0]

num_sample =10000
num_sample_test = 2000
batch_size = 32
epoch = 20
learning_rate=0.001

def weight_init(shape):
    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.01))

def bias_init(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))


def conv2d(input,weight,bias):
    input = tf.nn.conv2d(input,filter=weight,strides=[1,1,1,1],padding='SAME')
    return tf.add(input,bias)

def maxpool2_2(input):
    return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



x = tf.placeholder('float',[None,14*4*101*101],name='x')
y = tf.placeholder('float',[None,1],name='y')
lr = tf.placeholder('float',name='lr')
x_input = x / 255.

x_reshape = tf.reshape(x_input,[-1,101,101,14*4])

####conv
weight_conv = weight_init([3,3,14*4,128])
bias_conv = bias_init([128])
conv = conv2d(x_reshape,weight_conv,bias_conv)
x_relu = tf.nn.relu(conv)
x_maxpool= maxpool2_2(x_relu)

####conv
weight_conv = weight_init([3,3,128,64])
bias_conv = bias_init([64])
conv = conv2d(x_maxpool,weight_conv,bias_conv)
x_relu2 = tf.nn.relu(conv)
x_maxpool2= maxpool2_2(x_relu2)

####conv
weight_conv = weight_init([3,3,64,32])
bias_conv = bias_init([32])
conv = conv2d(x_maxpool2,weight_conv,bias_conv)
x_relu3 = tf.nn.relu(conv)
# x= maxpool2_2(x)

x_reshape= tf.reshape(x_relu3,[-1,32*26*26])

weight = weight_init([32*26*26,64])
bias = bias_init([64])

x_add = tf.add(tf.matmul(x_reshape,weight),bias)

x_relu4 = tf.nn.relu(x_add)
weight_last = weight_init([64,1])
bias = bias_init([1])

y_ = tf.add(tf.matmul(x_relu4,weight_last),bias)

loss_1 = tf.reduce_mean(tf.pow((y-y_),2))+tf.nn.l2_loss(weight_last)
loss = tf.pow(loss_1,0.5)

optimizer = tf.train.AdamOptimizer(learning_rate=0.05)

train = optimizer.minimize(loss_1)

init = tf.initialize_all_variables()



from data_preprocess.cikm import SubLineReader
from data_preprocess.read_file import readFile

with tf.Session() as sess:
    sess.run(init)

    for e in xrange(epoch):
        reader = SubLineReader(root + "train.txt", batch_size=batch_size)
        for i in xrange(num_sample//batch_size):
            try:
                print "========================="
                x_batch,y_batch = reader.next()
                # x_batch = np.zeros([100,15*4*101*101])
                # y_batch = np.zeros([100,1])
                x_batch = np.reshape(x_batch,[-1,14*4*101*101])
                y_batch = np.reshape(y_batch,[-1,1])
                # pdb.set_trace()
                [_,loss_,y_hat ,loss_1_] = sess.run([train,loss,y_,loss_1],feed_dict={x:x_batch,y:y_batch,lr:learning_rate})
                print "epoch:",e,"i/batch:",i,"/",num_sample//batch_size,"loss:",loss_

            except Exception,e:
                print Exception,":",e
                break

    result = []

    with open("/home/zsc/下载/data_new (3)/CIKM2017_testA/testA.txt", 'r') as f:
        i=0
        for line in f.readlines():
            if i%100 ==0:
                print i
            i+=1
            line = line.strip()
            lis = line.split(",")
            data = map(np.float32, lis[2].split())
            x_train_sub = []
            data = np.reshape(data, [-1, 15, 4, 101, 101])
            for i in xrange(14):
                x_train_s = data[:, i + 1, :, :, :] - data[:, i, :, :, :]
                # print x_train_s.shape
                x_train_sub.append(x_train_s)
            data= np.array(x_train_sub).flatten()

            x_batch = np.reshape(data, [-1, 15 * 4 * 101 * 101])
            predict = sess.run(y_, feed_dict={x: x_batch})
            # print predict
            result.append(predict[0][0])
            # data.append(line)
    # data = np.array(data, dtype='float32')

    f = open("result.csv", 'wb')
    for line in result:
        if line<0:
            line=0
        f.write(str(line)+'\n')
    f.close()
