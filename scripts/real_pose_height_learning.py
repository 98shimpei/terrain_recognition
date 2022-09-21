#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import random
import math
import cv2
import time
import sys
import cnn_models
import os


args = sys.argv
if len(args) == 1:
    args.append(1000)
if len(args) == 2:
    args.append(50)
if len(args) == 3:
    args.append("fwv")
print(args)

terrains_x = []
terrains_y = []
pose_terrains = []
surfaces = []
for filename in os.listdir("../terrains/x"):
    terrains_x.append(np.loadtxt("../terrains/x/"+filename, delimiter=","))
    terrains_y.append(np.loadtxt("../terrains/y/"+filename, delimiter=","))
for filename in os.listdir("../surfaces"):
    surfaces.append(np.loadtxt("../surfaces/"+filename, delimiter=","))
steppable_terrains = [terrains_x[i] for i in range(len(terrains_y)) if np.sum(terrains_y[i])==25]
for filename in os.listdir("../terrains/pose"):
    pose_terrains.append(np.loadtxt("../terrains/pose/"+filename, delimiter=","))

#----------------------------
# データの作成
# 画像サイズ（高さ，幅，チャネル数）
H, W, C = 21, 21, 1

pitch = np.zeros((H, W, 1))
roll = np.zeros((H, W, 1))
scale = np.ones((H, W, 1))

for i in range(W):
    for j in range(W):
        pitch[i, j, 0] = (j-H/2.0)*0.01
        roll[i, j, 0]  = (i-W/2.0)*0.01

def randomize(x, y):
    randomizer = np.arange(x.shape[0])
    np.random.shuffle(randomizer)
    return x[randomizer], y[randomizer]

def generate_data(num):
    x_data = np.zeros((num, H, W, C))
    y_data = np.zeros((num, 1, 1, 3))
    for i in range(num):
        surface_index = math.floor(random.random()*len(surfaces))
        surface_x = math.floor(random.random() * (surfaces[surface_index].shape[1] - 21))
        surface_y = math.floor(random.random() * (surfaces[surface_index].shape[0] - 21))
        x_data[i, :, :, 0] += surfaces[surface_index][surface_y:surface_y+21, surface_x:surface_x+21]
        
        #terrain
        if i < num*1.0:
            x_data[i, :, :, 0] += steppable_terrains[math.floor(random.random()*len(steppable_terrains))][13:34,13:34]
            for j in range(2):
                h = random.random()*0.3
                l = math.floor(random.random()*8+1)
                tmp = random.random()
                if tmp < 0.1: #端を消す
                    x_data[i, 0:l, :, 0] -= h * np.ones((l, x_data.shape[2]))
                elif tmp < 0.2:
                    x_data[i, x_data.shape[1]-l:x_data.shape[1], :, 0] -= h * np.ones((l, x_data.shape[2]))
                elif tmp < 0.3:
                    x_data[i, :, 0:l, 0] -= h * np.ones((x_data.shape[1], l))
                elif tmp < 0.4:
                    x_data[i, :, x_data.shape[2]-l:x_data.shape[2], 0] -= h * np.ones((x_data.shape[1], l))
        else:
            x_data[i, :, :, 0] += pose_terrains[math.floor(random.random()*len(pose_terrains))][13:34,13:34]

        #rotate
        theta = random.random() * 360
        M = cv2.getRotationMatrix2D((math.floor(x_data.shape[2] / 2.), math.floor(x_data.shape[1] / 2.)), theta, 1)
        x_data[i, :, :, 0] = cv2.warpAffine(x_data[i], M, (x_data.shape[2], x_data.shape[1]))

        #tilt
        p = 2.0*random.random() - 1.0
        r = 2.0*random.random() - 1.0
        s = 1.0*random.random() - 0.5
        x_data[i] += p * pitch + r * roll + s * scale + 0.05*np.random.rand(H, W, 1) - 0.025 * np.ones((H, W, 1))
        y_data[i, 0, 0] = np.array([p, r, s])
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

#----------------------------
# 学習
if "w" in args[3]:
    model_pose = cnn_models.cnn_pose((H,W,C), "../checkpoints/checkpoint")
else:
    model_pose = cnn_models.cnn_pose((H,W,C), "")
if "v" in args[3]:
    model_pose.summary()
if "f" in args[3]:
    model_pose.fit(x_train, y_train[:,:,:,:2], batch_size=100, epochs=int(args[2]))
    model_pose.save_weights('../checkpoints/checkpoint/checkpoint_pose')


if "w" in args[3]:
    model_height = cnn_models.cnn_height((H,W,C), "../checkpoints/checkpoint")
else:
    model_height = cnn_models.cnn_height((H,W,C), "")
if "v" in args[3]:
    model_height.summary()
if "f" in args[3]:
    model_height.fit(x_train, y_train[:,:,:,2], batch_size=100, epochs=int(args[2]))
    model_height.save_weights('../checkpoints/checkpoint/checkpoint_height')
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
    cv2.waitKey(1)

    print(len(steppable_terrains))

    for i in range(5):
        print("answer")
        print(y_train[i])
        print("predict")
        print(model_pose.predict(x_train[i:i+1]))
        print(model_height.predict(x_train[i:i+1]))
        cv2.imshow('test',x_train[i] * 1.0 + 0.5)
        cv2.waitKey(1)
        time.sleep(5)
        print("---")

    print("answer")
    print(y_test[70])
    print("predict")
    print(model_pose.predict(x_test[70:71]))
    print(model_height.predict(x_test[70:71]))
    cv2.imshow('test',x_test[70] * 1.0 + 0.5)
    cv2.waitKey(1)
    time.sleep(5)
    print("---")
