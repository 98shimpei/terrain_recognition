#!/usr/bin/env python3
#############################
# tensorflow2のCNNの実装例1（もっとも素人ぽい）
# Sequential APIを用いる場合
#############################
import tensorflow as tf
import numpy as np
import random
import math
import cv2
import time
import sys


args = sys.argv
if len(args) == 1:
    args.append(1000)
if len(args) == 2:
    args.append(50)
if len(args) == 3:
    args.append("fwv")
print(args)

#----------------------------
# データの作成
# 画像サイズ（高さ，幅，チャネル数）
H, W, C = 33, 33, 1

# MNISTデータの読み込み
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#print(x_train.shape)
#print(x_train[0,0,1])

pitch = np.zeros((H*2-1, W*2-1, 1))
roll = np.zeros((H*2-1, W*2-1, 1))
scale = np.ones((H*2-1, W*2-1, 1))

for i in range(H*2-1):
    for j in range(W*2-1):
        pitch[i, j, 0] = (j-(H-1))*0.01
        roll[i, j, 0]  = (i-(W-1))*0.01

def randomize(x, y):
    randomizer = np.arange(x.shape[0])
    np.random.shuffle(randomizer)
    return x[randomizer], y[randomizer]

def generate_data(num):
    x_data = np.ones((num, H*2-1, W*2-1, C))
    y_data = np.ones((num, 1, 1, 1))
    for i in range(num):
        p = 2.0*random.random() - 1.0
        r = 2.0*random.random() - 1.0
        s = 1.0*random.random() - 0.5
        x_data[i] = p * pitch + r * roll + s * scale + 0.05*np.random.rand(H*2-1, W*2-1, 1) - 0.025 * np.ones((H*2-1, W*2-1, 1))
        y_data[i, 0, 0] = 1
    x_data, y_data = randomize(x_data, y_data)

    for i in range(math.floor(num*0.7)):
        begin, end = np.sort([math.floor(random.random()*(H+1)), math.floor(random.random()*(H+1))])

        l = end - begin
        if l == 0 :
            continue
        h = random.random() * 0.6 - 0.3
        x_data[i, begin : end, :, 0] += np.ones((l, x_data.shape[2])) * h
        if h > 0.02:
            if begin > 22 or end < 43:
                y_data[i, 0, 0] = 0
        elif h < -0.02:
            if not (end <= 22 or begin >= 43 or (begin >= 26 and end <=39)):
                y_data[i, 0, 0] = 0
    for i in range(num):
        theta = random.random() * 60 + 60
        M = cv2.getRotationMatrix2D((math.floor(x_data.shape[2] / 2.), math.floor(x_data.shape[1] / 2.)), theta, 1)
        x_data[i, :, :, 0] = cv2.warpAffine(x_data[i], M, (x_data.shape[2], x_data.shape[1]))
    x_data, y_data = randomize(x_data, y_data)

    for i in range(math.floor(num*0.7)):
        begin, end = np.sort([math.floor(random.random()*(H+1)), math.floor(random.random()*(H+1))])

        l = end - begin
        if l == 0 :
            continue
        h = random.random() * 0.6 - 0.3
        x_data[i, begin : end, :, 0] += np.ones((l, x_data.shape[2])) * h
        if h > 0.02:
            if begin > 22 or end < 43:
                y_data[i, 0, 0] = 0
        elif h < -0.02:
            if not (end <= 22 or begin >= 43 or (begin >= 26 and end <=39)):
                y_data[i, 0, 0] = 0
    for i in range(num):
        theta = random.random() * 180
        M = cv2.getRotationMatrix2D((math.floor(x_data.shape[2] / 2.), math.floor(x_data.shape[1] / 2.)), theta, 1)
        x_data[i, :, :, 0] = cv2.warpAffine(x_data[i], M, (x_data.shape[2], x_data.shape[1]))
    x_data, y_data = randomize(x_data, y_data)

    x_data = x_data[:, math.floor(H/2):x_data.shape[1]-math.floor(H/2), math.floor(W/2):x_data.shape[2]-math.floor(W/2)]

    #for i in range(math.floor(num*0.8)):
    #    n = math.floor(7 * random.random())
    #    for j in range(n):
    #        l = math.floor(5 * random.random()) + 2
    #        x = math.floor((W-l) * random.random())
    #        y = math.floor((H-l) * random.random())
    #        h = 0.5 * random.random()
    #        x_data[i, x:x+l, y:y+l] -= h * np.ones((l, l, 1))
    #        pp = 1.0 * random.random()
    #        qq = 1.0 * random.random()
    #        for p in range(l):
    #            for q in range(l):
    #                x_data[i, x+p, y+q, 0] -= (p * pp + q * qq) * 0.01
    #x_data, y_data = randomize(x_data, y_data)
    #for i in range(math.floor(num*0.1)):
    #    l = math.floor(random.random()*6)+1
    #    x_data[i, 0 : l, :, 0] -= np.ones((l, x_data.shape[2])) * random.random() * 0.4
    #x_data, y_data = randomize(x_data, y_data)
    #for i in range(math.floor(num*0.1)):
    #    l = math.floor(random.random()*6)+1
    #    x_data[i, x_data.shape[1]-l : x_data.shape[1], :, 0] -= np.ones((l, x_data.shape[2])) * random.random() * 0.4
    #x_data, y_data = randomize(x_data, y_data)
    #for i in range(math.floor(num*0.1)):
    #    l = math.floor(random.random()*6)+1
    #    x_data[i, :, 0 : l, 0] -= np.ones((x_data.shape[1], l)) * random.random() * 0.4
    #x_data, y_data = randomize(x_data, y_data)
    #for i in range(math.floor(num*0.1)):
    #    l = math.floor(random.random()*6)+1
    #    x_data[i, :, x_data.shape[2]-l : x_data.shape[2], 0] -= np.ones((x_data.shape[1], l)) * random.random() * 0.4

    return x_data, y_data

x_train, y_train = generate_data(int(args[1]))
x_test, y_test = generate_data(100)

# 画像の正規化
#x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255

# （データ数，高さ，幅，チャネル数）にrehspae
#x_train = x_train.reshape(x_train.shape[0], H, W, C)
#x_test = x_test.reshape(x_test.shape[0], H, W, C)
#y_train = y_train.reshape(y_train.shape[0], 1, 1, 3)
#y_test = y_test.reshape(y_test.shape[0], 1, 1, 3)
#----------------------------

#----------------------------
# Sequentialを用いたネットワークの定義
# - addメソッドを用いてlayerインスタンス（Conv2D，BatchNormalization，ReLU，MaxPooling2D，Flatten，Dense，Dropoutなど）をSequentialに追加していく
# - compileメソッドを用いて，最適化方法（adam），損失関数（sparse_categorical_crossentropy），評価方法（accuracy）を設定
def cnn_steppable_region(input_shape):
    model = tf.keras.models.Sequential()

    # conv1
    model.add(tf.keras.layers.Conv2D(32, (11, 11), input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    #model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # conv2
    model.add(tf.keras.layers.Conv2D(32, (9, 9)))
    #model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())    
    #model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # conv3
    model.add(tf.keras.layers.Conv2D(32, (9, 9)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # conv3
    model.add(tf.keras.layers.Conv2D(32, (7, 7)))
    model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.ReLU())
    #model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    ## fc1
    #model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dense(3, activation='linear'))
    #model.add(tf.keras.layers.Dropout(0.2))

    # fc2
    #model.add(tf.keras.layers.Dense(4))
    #model.add(tf.keras.layers.Dense(2, activation='softmax'))

    # 学習方法の設定
    #model.compile(optimizer='adam',loss='mse',metrics=['mae', 'mse'])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    if "w" in args[3]:
        model.load_weights('./checkpoint/checkpoint_steppable_region')

    return model

#----------------------------

#----------------------------
# 学習
# - cnn関数を実行しネットワークを定義
# - fitで学習を実行
model_steppable_region = cnn_steppable_region((H,W,C))
if "v" in args[3]:
    model_steppable_region.summary()
if "f" in args[3]:
    model_steppable_region.fit(x_train, y_train, batch_size=100, epochs=int(args[2]))
    model_steppable_region.save_weights('./checkpoint/checkpoint_steppable_region')

#----------------------------

#----------------------------
# 学習データに対する評価
##train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
#train_loss, train_mae, train_mse = model_height.evaluate(x_train, y_train[:,:,:,2], verbose=2)
#print('Train data loss:', train_loss)
##print('Train data accuracy:', train_accuracy)
#print("Testing set Mean Abs Error: {:5.2f} MPG".format(train_mae))

#----------------------------

#----------------------------
## 評価データに対する評価
##test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
##print('Test data loss:', test_loss)
##print('Test data accuracy:', test_accuracy)
#test_loss, test_mae, test_mse = model.evaluate(x_test, y_test, verbose=2)
#print('Test data loss:', test_loss)
#print("Testing set Mean Abs Error: {:5.2f} MPG".format(test_mae))
#
#print("answer")
#print(y_test[0])
#print("predict")
#print(model.predict(x_test[:1]))


if "v" in args[3]:
    cv2.imshow('test',x_test[0] * 1.0 + 0.5)

    for i in range(2):
        print("answer")
        print(y_train[i])
        print("predict")
        print(np.argmax(model_steppable_region.predict(x_train[i:i+1]), axis=3))
        cv2.imshow('test',x_train[i] * 0.7 + 0.5)
        cv2.waitKey(1)
        time.sleep(2)
        print("---")

    print("answer")
    print(y_test[70])
    print("predict")
    print(np.argmax(model_steppable_region.predict(x_test[70:71]), axis=3))
    cv2.imshow('test',x_test[70] * 0.7 + 0.5)
    cv2.waitKey(1)
    time.sleep(5)
    print("---")

    #x_data = np.ones((1, H, W, C))
    #y_data = np.ones((1, 1, 1, 1))
    #for i in range(1):
    #    p = 2.0*random.random() - 1.0
    #    r = 2.0*random.random() - 1.0
    #    s = 1.0*random.random() - 0.5
    #    x_data[i] = p * pitch + r * roll + s * scale + 0.05*np.random.rand(H, W, 1) - 0.025 * np.ones((H, W, 1))
    #    y_data[i, 0, 0] = 1
    #for i in range(1):
    #    begin = 15
    #    end = 20
    #    l = end - begin
    #    h = -0.2
    #    x_data[i, begin : end, :, 0] += np.ones((l, x_data.shape[2])) * h
    #    if h > 0.02:
    #        if begin >= 6 or end <= 27:
    #            y_data[i, 0, 0] = 0
    #    elif h < -0.02:
    #        if not (end <= 6 or begin >= 27 or (begin >= 10 and end <=23)):
    #            y_data[i, 0, 0] = 0
    #print("answer")
    #print(y_data)
    #print("predict")
    #print(model_steppable_region.predict(x_data))
    #cv2.imshow('test',x_data[0] * 0.7 + 0.5)
    #cv2.waitKey(1)
    #time.sleep(5)
    #print("---")

#print("answer")
#print(y_test[71])
#print("predict")
#tmp = model_height.predict(x_test[71:72])[0, 0, 0, 0]
#x_test[71] -= tmp * scale
#print(tmp)
#print(model_pose.predict(x_test[71:72]))
#cv2.imshow('test',x_test[71] * 1.0 + 0.5)
#cv2.waitKey(1)
#time.sleep(5)
#print("---")
#
#x_final = np.zeros((1, H, W, C))
##p = 0.3
##r = -0.1
##s = 0.3
##x_final[0] = p * pitch + r * roll + s * scale + 0.00*np.random.rand(H, W, 1)
##y_final = np.array([p, r, s])
#for j in range(W):
#    for i in range(10):
#        x_final[0, i, j, 0] = -0.1 + i * 0.003 + j * 0.001
#    for i in range(10, 15):
#        x_final[0, i, j, 0] = -0.09 + j * 0.001
#    for i in range(15, 25):
#        x_final[0, i, j, 0] = -0.1 + (24 - i) * 0.003 + j * 0.001
#y_final = np.array([0, 0.1, -0.13])
#
#
#print("answer")
#print(y_final)
#print("predict")
#tmp = model_height.predict(x_final)[0, 0, 0, 0]
#x_final -= tmp * scale
#print(tmp)
#print(model_pose.predict(x_final))
#cv2.imshow('test',x_final[0] * 1.0 + 0.5)
#cv2.waitKey(1)
#time.sleep(10)
#----------------------------
