#coding=utf-8
import numpy as np
import os
import sys
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
reload(sys)
sys.setdefaultencoding('utf-8')

radius= 30
maps = 60
center_width = 2*radius
center_hight = 2*radius


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
            right = 51 + center_radius
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

reader = CenterFeatureReader("/home/zsc/下载/data_new/CIKM2017_train/train.txt",center_radius=radius,batch_size=4000)
X,y = reader.next()

# X = X / 255.
# y = y / 100.

X = np.reshape(X, [-1, maps * center_width * center_hight])

index = [i for i in range(X.shape[0])]
np.random.shuffle(index)
X = X[index]
y = y[index]

x_train,x_test,y_train,y_test  = train_test_split(X,y,test_size=0.05)

#train the model
print("train lr model...")
linear = Lasso(normalize=True,alpha=0.1)
linear = linear.fit(x_train,y_train)
print("train lr model...end ")
gdbr = GradientBoostingRegressor(n_estimators=100)
gdbr = gdbr.fit(x_train,y_train)
print("train gb model... over")
#test the model

y_pred_lr = linear.predict(x_test)
y_pred_gb = gdbr.predict(x_test)

y_pred = (y_pred_gb+y_pred_lr)/2.0

loss = np.mean(np.square((y_test-y_pred)))
loss = np.power(loss,0.5)

# loss = mean_squared_error(y_test,y_pred)
print loss

# sys.exit()



result = []

with open("/home/zsc/下载/data_new (3)/CIKM2017_testA/testA.txt", 'r') as f:
    i=0
    while(1):
        line = f.readline()
        line = line.strip()

        if line=="" or line is None:
            break
        lis = line.split(",")
        i+=1
        if i%100==0:
            print i
        #preprocess
        left = 51 - radius
        right = 51 + radius
        X_train = map(float, lis[2].split(' '))
        x_train = np.reshape(X_train, [-1, 60, 101, 101])
        x_train_feature = x_train[:, :, left:right, left:right]
        # print x_train_feature.flatten()
        x_train_feature = np.reshape(x_train_feature, [-1, maps * center_width * center_hight])
        # x_train_feature = x_train_feature/255.
        y_pred_lr = linear.predict(x_train_feature)
        y_pred_gb = gdbr.predict(x_train_feature)

        y_pred = (y_pred_gb + y_pred_lr) / 2.0
        # print predict
        result.append(y_pred[0])
        # data.append(line)
# data = np.array(data, dtype='float32')

f = open("result.csv", 'wb')
for line in result:
    if line<0:
        line=0
    f.write(str(line)+'\n')
f.close()
