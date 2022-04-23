import glob
import pandas as pd
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras import *
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, Dropout, Bidirectional,Conv1D

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

if not os.path.exists("d.joblib"):
    tmp = glob.glob("dataall/*")
    d = dict(zip(tmp,list(range(len(tmp)))))
    joblib.dump(d,"d.joblib")
else:
    d = joblib.load("d.joblib")
    #d是每个监测点文件及其index的字典


# # 划分时间步制作数据集


def make_train_data(tmp):
    train_data = []
    label = []
    time = []
    time_step=5
    for i in range(time_step,tmp.shape[0]-time_step):
        train_data.append(tmp.loc[i-time_step:i-1,["sumall"]].values)
        label.append(tmp.loc[i,["sumall"]].values)
        time.append(tmp.loc[i,["time"]].values)
    return train_data,label,time


def onehot(num):
    index = [0]*190
    index[num] = 1
    return index

# # 制作模型输入和输出

trainall=[]
labelall=[]
vdid=[]
timeall=[]

for i in tqdm(glob.glob("dataall/*")):
    tmp = pd.read_csv(i)
    train_data, label, time = make_train_data(tmp)
    trainall.extend(train_data)
    labelall.extend(label)
    timeall.extend(time)
    vdid.extend([onehot(d[i])]*len(time)) # 将监测点的名称保存为190维数列的1值


# # 按照百分之20测试集切分数据集


trainall=np.array(trainall)

labelall=np.array(labelall)

vdid=np.array(vdid)

trainall = trainall.astype('float64')
labelall=labelall.astype('float64')
vdid=vdid.astype('float64')

timeall=np.array(timeall)

np.random.seed(24)

index=np.random.permutation(range(trainall.shape[0]))

trainall=trainall[index]

labelall=labelall[index]

vdid=vdid[index]

timeall=timeall[index]

allsize=trainall.shape[0]

testsize=0.2

train_data=trainall[int(allsize*testsize):]
val_data=trainall[:int(allsize*testsize)]

train_vdid=vdid[int(allsize*testsize):]
val_vdid=vdid[:int(allsize*testsize)]

train_y=labelall[int(allsize*testsize):]
val_y=labelall[:int(allsize*testsize)]


train_time=timeall[int(allsize*testsize):]
test_time=timeall[:int(allsize*testsize)]

#构建模型结构
def get_model(lstmdim=64,cell="BILSTM"):
    global input2
    inputs = Input(shape=(train_data.shape[1], train_data.shape[2])) #模型的入口,设置输入的维度
    input_vdid=Input(shape=(190, ))         #输入为190维向量
    outvdid=Dense(12)(input_vdid)
    if cell=="BILSTM":
        input2 = Bidirectional(LSTM(lstmdim, return_sequences=False))(inputs)   # 双向lstm层，return_sequences=False只返回最后时间步的输出，隐藏层神经元维度为lstmdim
    elif cell=="BIGRU":
        input2 = Bidirectional(GRU(lstmdim, return_sequences=False))(inputs)   # 双向gru层，return_sequences=False只返回最后时间步的输出，隐藏层神经元维度为lstmdim
    x = Concatenate()([input2,outvdid])
    input2 = Dense(12, activation="relu")(x)  # 全联接层，神经元数量为12
    output = Dense(1, activation='relu')(input2)  # 全连接层，合并监测点和信息
    model = Model(inputs=[inputs,input_vdid], outputs=output)  # 告诉模型输入和输出是哪个
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])  # 定义损失函数标准差mse，优化器adam，评分标准 acc
    model.summary()  # 打印模型结构
    return model


# 训练LSTM


callbacks = [EarlyStopping(monitor='mse', verbose=1, patience=1000), # verbose=1为输出进度条记录
             ModelCheckpoint("bestmodel_lstm.hdf5", monitor='val_mse', # 回调函数将模型保存到文件中
                             mode='min', verbose=0, save_best_only=True,save_weights_only=True)]
# 提前结束训练避免过拟合，在模型测试集准确率最高的时候保存模型，并且只保存权重

cellname="BILSTM"
model=get_model(lstmdim=64,cell=cellname)


history=model.fit([train_data,train_vdid], train_y, epochs=30, batch_size=64, validation_data=([val_data,val_vdid], val_y) ,callbacks=callbacks)
model.load_weights("bestmodel_lstm.hdf5")


# # 绘制损失图

val_loss = history.history['val_mse']
loss = history.history['mse']
epochs = range(1, len(loss ) + 1)

plt.title('lstm_mse')
plt.plot(epochs, loss, 'red', label='Training mse') # 训练误差
plt.plot(epochs, val_loss, 'blue', label='Validation mse') # 测试误差
plt.legend()
plt.show()


# 计算指标
testpre=model.predict([val_data,val_vdid])
lstm_testpre=np.squeeze(testpre)
y_test1=np.squeeze(val_y)
print ("lstm_mae",mean_absolute_error(np.array(y_test1),lstm_testpre)) # 均方误差
print ("lstm_mse",mean_squared_error(np.array(y_test1),lstm_testpre)) # 均方根误差
print ("lstm_rmse",np.sqrt(mean_squared_error(np.array(y_test1),lstm_testpre))) # 平均绝对误差
print ("lstm_r2_score",r2_score(np.array(y_test1),np.squeeze(lstm_testpre))) # 决定系数【0，1】，越大越好


# 训练gru


cellname="BIGRU"
model=get_model(lstmdim=64,cell=cellname)

callbacks = [EarlyStopping(monitor='mse', verbose=1, patience=1000),
             ModelCheckpoint("bestmodel_gru.hdf5", monitor='val_mse',
                             mode='min', verbose=0, save_best_only=True,save_weights_only=True)] # 在模型测试集准确率最高的时候保存模型，并且只保存权重

history=model.fit([train_data,train_vdid], train_y, epochs=30, batch_size=64, validation_data=([val_data,val_vdid], val_y) ,callbacks=callbacks)
model.load_weights("bestmodel_gru.hdf5")


# 损失图


val_loss = history.history['val_mse']
loss = history.history['mse']
epochs = range(1, len(loss ) + 1)

plt.title('gru_mse')
plt.plot(epochs, loss, 'red', label='Training mse')
plt.plot(epochs, val_loss, 'blue', label='Validation mse')
plt.legend()
plt.show()


# 评估


testpre = model.predict([val_data, val_vdid])
gru_testpre = np.squeeze(testpre)
y_test1 = np.squeeze(val_y)
print ("gru_mae",mean_absolute_error(np.array(y_test1),gru_testpre))
print ("gru_mse",mean_squared_error(np.array(y_test1),gru_testpre))
print ("gru_rmse",np.sqrt(mean_squared_error(np.array(y_test1),gru_testpre)))
print ("gru_r2_score",r2_score(np.array(y_test1),np.squeeze(gru_testpre)))


# 绘制总体预测图


import matplotlib.pyplot as plt
plt.figure(figsize=(30,10))
huatu_ypre=testpre
huatu_ytest=y_test1
snum=100
enum=200
plt.title('traffic flow prediction')
plt.plot(list(range(lstm_testpre[snum:enum][:].shape[0])), lstm_testpre[snum:enum][:], 'red', label='lstm_pre')
plt.plot(list(range(lstm_testpre[snum:enum][:].shape[0])), gru_testpre[snum:enum][:], 'black', label='gru_pre')
plt.plot(list(range(huatu_ytest[snum:enum][:].shape[0])), huatu_ytest[snum:enum][:], 'blue', label='true')
plt.legend()
plt.savefig('prediction.png')
plt.show()