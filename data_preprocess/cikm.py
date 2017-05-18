#coding=utf-8
import numpy as np



def LineReader(filename,batch_size=1) :
    datas = []
    labels = []
    with open(filename) as f:
        # for line in f.readlines():
        while (1):
            line = f.readline()
            if line == "" or line is None:
                break
            line = line.strip().split(',')
            print line[0]
            X_train = map(float, line[2].split(' '))
            x_train = np.reshape(X_train, [-1, 60, 101, 101])
            datas.append(x_train)
            labels.append(float(line[1]))
            if len(labels) >= batch_size:
                yield np.array(datas), np.array(labels)
                datas = []
                labels = []
    if len(labels) > 0:
        yield np.array(datas), np.array(labels)

def SubLineReader(filename,batch_size=1) :
    datas = []
    labels = []
    with open(filename) as f:
        # for line in f.readlines():
        while (1):
            line = f.readline()
            if line == "" or line is None:
                break
            line = line.strip().split(',')
            X_train = map(float, line[2].split(' '))
            x_train = np.reshape(X_train, [-1, 15,4, 101, 101])
            x_train_sub = []
            for i in xrange(14):
                x_train_s = x_train[:,i+1,:,:,:] - x_train[:,i,:,:,:]
                # print x_train_s.shape
                x_train_sub.append(x_train_s)
            x_train = np.array(x_train_sub).flatten()
            datas.append(x_train)
            labels.append(float(line[1]))
            if len(labels) >= batch_size:
                yield np.array(datas), np.array(labels)
                datas = []
                labels = []
    if len(labels) > 0:
        yield np.array(datas), np.array(labels)




def CenterFeatureReader(filename,center_radius,batch_size=1) :
    datas=[]
    labels=[]
    with open(filename) as f:
        # for line in f.readlines():
        while(1):
            line = f.readline()
            if line=="" or line is None:
                break
            line = line.strip().split(',')


            # 15 *4*101*101
            # 51-center_radius  -  51+ center_radius
            left = 51 - center_radius
            right = 51 + center_radius+1
            X_train = map(float, line[2].split(' '))
            x_train = np.reshape(X_train, [-1, 60, 101, 101])
            x_train_feature = x_train[:, :, left:right, left:right]

            # print "center:",x_train[:,:,51,51],"y:",line[1]
            datas.append(x_train_feature)
            labels.append(float(line[1]))
            if len(labels)>=batch_size:
                yield np.array(datas), np.array(labels)
                datas=[]
                labels=[]
    if len(labels)>0:
        yield np.array(datas), np.array(labels)

'''
#visiualize
root="/home/zsc/下载/data_new/CIKM2017_train/"
reader = CenterFeatureReader(root+"/train.txt",center_radius=0,batch_size=1000)
x,y = reader.next()
x = x[:,:,-1,:,:].flatten()



from pylab import *

import matplotlib.pyplot as plt

# basic
f1 = plt.figure(1)
# plt.subplot(211)
plt.scatter(x,y)
plt.show()
'''
