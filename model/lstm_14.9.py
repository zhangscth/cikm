#coding=utf-8
from data_preprocess import read_file
import  tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pdb


root="/home/zsc/下载/data_new/CIKM2017_train/"
# print data[0]


num_sample =10000
num_sample_test = 2000
batch_size = 32
epoch = 20
learning_rate=0.001




time_step = 60
lstm_hidden = 128



x = tf.placeholder('float',[None,15*4*101*101])
y = tf.placeholder('float',[None,1])
lr = tf.placeholder('float')
x_scale = x / 255.


x_split = tf.split(x_scale,time_step,axis=1)


# Define lstm cells with tensorflow
# Forward direction cell
lstm_cell = rnn.BasicLSTMCell(lstm_hidden, forget_bias=1.0)
'''
lstm_fw_cell = rnn.BasicLSTMCell(lstm_hidden, forget_bias=1.0)
# Backward direction cell
lstm_bw_cell = rnn.BasicLSTMCell(lstm_hidden, forget_bias=1.0)

# **步骤5：用全零来初始化state
init_state_forw = lstm_fw_cell.zero_state(batch_size, dtype=tf.float32)
init_state_back = lstm_bw_cell.zero_state(batch_size, dtype=tf.float32)
'''

state = lstm_cell.zero_state(batch_size,dtype=tf.float32)

'''
outputs = list()
with tf.variable_scope('RNN'):
    for timestep in range(time_step):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        # 这里的state保存了每一层 LSTM 的状态
        cell_output, state = lstm_cell(x_split[timestep], state)
        outputs.append(cell_output)
    # (outputs, output_state_fw, output_state_bw) = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_split,
    #                                 initial_state_fw=init_state_forw,
    #                                 initial_state_bw= init_state_back,sequence_length=batch_len)

'''

rnn_cell = rnn.BasicLSTMCell(lstm_hidden)
outputs,state = rnn.static_rnn(rnn_cell,x_split,dtype='float32')

weight = tf.Variable(tf.truncated_normal([lstm_hidden,1],mean=0,stddev=1e-8))
bias = tf.Variable(tf.constant([0.0]))


h_state = outputs[-1]

y_ = tf.matmul(h_state,weight)+bias

loss_1 = tf.reduce_mean(tf.pow((y-y_),2)) + tf.reduce_sum(tf.square(weight)) * 0.01
loss = tf.pow(loss_1,0.5)

optimizer = tf.train.AdamOptimizer(learning_rate=0.05)

train = optimizer.minimize(loss_1)

init = tf.initialize_all_variables()



from data_preprocess.cikm import LineReader
from data_preprocess.read_file import readFile

with tf.Session() as sess:
    sess.run(init)
    for e in xrange(epoch):
        reader = LineReader(root + "train.txt", batch_size=batch_size)
        for i in xrange(num_sample//batch_size):

            print "========================="
            x_batch,y_batch = reader.next()
            x_batch = np.reshape(x_batch,[-1,15*4*101*101])
            y_batch = np.reshape(y_batch,[-1,1])

            sample_num = y_batch.shape[0]
            # pdb.set_trace()
            [_,loss_,y_hat ,loss_1_] = sess.run([train,loss,y_,loss_1],feed_dict={x:x_batch,y:y_batch,lr:learning_rate})
            print "epoch:",e," loss",loss_


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
            x_train_feature = np.reshape(x_train, [-1, 15*4 * 101* 101])
            sample_num = x_train_feature.shape[0]
            predict = sess.run(y_, feed_dict={x: x_train_feature})
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
