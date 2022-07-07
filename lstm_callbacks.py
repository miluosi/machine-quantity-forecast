import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (20, 10) # 单位是inches
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd

seed = 7
batch_size = 1
epochs = 50
footer = 3
look_back=1

def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        x = dataset[i: i + look_back, 0]
        dataX.append(x)
        y = dataset[i + look_back, 0]
        dataY.append(y)
        print('X: %s, Y: %s' % (x, y))
    return np.array(dataX), np.array(dataY)

def build_model():
    model = Sequential()
    model.add(LSTM(units=4, input_shape=(look_back, 1)))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

if __name__ == '__main__':

    # 设置随机种子
    np.random.seed(seed)

    # 导入数据
    data = pd.read_excel(r'data1.xlsx')
    #data = read_csv(filename, usecols=[1], engine='python', skipfooter=footer)
    #dataset = data.values.astype('float32')
    # 标准化数据
    scaler = MinMaxScaler()
    dataset = data.iloc[:, 1]
    dataset = np.array(dataset).reshape(-1, 1)
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * 0.9)
    validation_size = len(dataset) - train_size
    train, validation = dataset[0: train_size, :], dataset[train_size: len(dataset), :]

    # 创建dataset，让数据产生相关性
    X_train, y_train = create_dataset(train)
    X_validation, y_validation = create_dataset(validation)

    # 将输入转化成为【sample， time steps, feature]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

    # 训练模型
    model = build_model()
    train_loss = []
    val_loss = []
    # model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    from keras.callbacks import ModelCheckpoint

    checkpointer = ModelCheckpoint(filepath='weights.best.hdf5',
                                   monitor='val_loss',
                                   save_best_only=True,
                                   verbose=2)
    history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size, verbose=2,
                        validation_data=(X_validation, y_validation),
                        callbacks=[checkpointer])
    print("train loss:", history.history['loss'][0])
    print("val loss:", history.history['val_loss'][0])
    plt.rcParams['figure.figsize'] = (20, 10)  # 单位是inches
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    model.save('my_model_2.h5')