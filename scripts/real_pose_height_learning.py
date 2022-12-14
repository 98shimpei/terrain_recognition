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
for filename in os.listdir("../terrains/pose"):
    pose_terrains.append(np.loadtxt("../terrains/pose/"+filename, delimiter=","))

#----------------------------
# データの作成
# 画像サイズ（高さ，幅，チャネル数）
H, W, C = 25, 25, 1

pitch = np.zeros((53, 53, 1))
roll = np.zeros((53, 53, 1))
scale = np.ones((53, 53, 1))

for i in range(53):
    for j in range(53):
        pitch[i, j, 0] = (j-26)*0.01
        roll[i, j, 0]  = (i-26)*0.01

def randomize(x, y):
    randomizer = np.arange(x.shape[0])
    np.random.shuffle(randomizer)
    return x[randomizer], y[randomizer]

def generate_data(num):
    x_data = np.zeros((num, 53, 53, 1))
    y_data = np.zeros((num, 1, 1, 3))
    for i in range(num):
        #terrain
        if i < num*1.0:
            terrain_index = math.floor(random.random()*len(terrains_x))
            while (terrains_y[terrain_index][5] < 0.5):
                terrain_index = math.floor(random.random()*len(terrains_x))
            x_data[i, :, :, 0] += terrains_x[terrain_index]
            y_tmp = terrains_y[terrain_index]
            y_n = np.array([y_tmp[1], y_tmp[2], y_tmp[3]])
            #for j in range(2):
            #    h = random.random()*0.3
            #    l = math.floor(random.random()*8+1)
            #    tmp = random.random()
            #    if tmp < 0.1: #端を消す
            #        x_data[i, 0:l, :, 0] -= h * np.ones((l, x_data.shape[2]))
            #    elif tmp < 0.2:
            #        x_data[i, x_data.shape[1]-l:x_data.shape[1], :, 0] -= h * np.ones((l, x_data.shape[2]))
            #    elif tmp < 0.3:
            #        x_data[i, :, 0:l, 0] -= h * np.ones((x_data.shape[1], l))
            #    elif tmp < 0.4:
            #        x_data[i, :, x_data.shape[2]-l:x_data.shape[2], 0] -= h * np.ones((x_data.shape[1], l))
        else:
            x_data[i, :, :, 0] += pose_terrains[math.floor(random.random()*len(pose_terrains))]

        #rotate
        theta = random.random() * 360
        M = cv2.getRotationMatrix2D((math.floor(x_data.shape[2] / 2.), math.floor(x_data.shape[1] / 2.)), theta, 1)
        x_data[i, :, :, 0] = cv2.warpAffine(x_data[i], M, (x_data.shape[2], x_data.shape[1]))
        R = np.array([[np.cos(np.deg2rad(-theta)), -np.sin(np.deg2rad(-theta))], [np.sin(np.deg2rad(-theta)), np.cos(np.deg2rad(-theta))]])
        y_n[:2] = np.dot(R, y_n[:2])

        #surface
        surface_index = math.floor(random.random()*len(surfaces))
        surface_x = math.floor(random.random() * (surfaces[surface_index].shape[1] - 25))
        surface_y = math.floor(random.random() * (surfaces[surface_index].shape[0] - 25))
        x_data[i, 14:39, 14:39, 0] += surfaces[surface_index][surface_y:surface_y+25, surface_x:surface_x+25]
        
        #tilt
        p = 2.0*random.random() - 1.0
        r = 2.0*random.random() - 1.0
        s = 1.0*random.random() - 0.5
        x_data[i] += p * pitch + r * roll + s * scale# + 0.05*np.random.rand(H, W, 1) - 0.025 * np.ones((H, W, 1))
        y_data[i, 0, 0] = np.array([p + (-y_n[0]/y_n[2]), r + (-y_n[1]/y_n[2]), s + y_tmp[4]])

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
    x_data = x_data[:,14:39,14:39]
    return x_data, y_data

x_train, y_train = generate_data(int(args[1]))
x_test, y_test = generate_data(100)

#----------------------------
# 学習
print("pose")
if "w" in args[3]:
    model_pose = cnn_models.cnn_pose((H,W,C), "../checkpoints/checkpoint")
else:
    model_pose = cnn_models.cnn_pose((H,W,C), "")
if "v" in args[3]:
    model_pose.summary()
if "f" in args[3]:
    model_pose.fit(x_train, y_train[:,:,:,:2], batch_size=100, epochs=int(args[2]))
    model_pose.save_weights('../checkpoints/checkpoint/checkpoint_pose')


print("height")
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
#train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
train_loss, train_mae, train_mse = model_height.evaluate(x_train, y_train[:,:,:,2], verbose=0)
print("height train mae: ", train_mae)

train_loss, train_mae, train_mse = model_pose.evaluate(x_train, y_train[:,:,:,:2], verbose=0)
print("pose  train mae: ", train_mae)

#----------------------------

#----------------------------
# 評価データに対する評価
test_loss, test_mae, test_mse = model_height.evaluate(x_test, y_test[:,:,:,2], verbose=0)
print("height train mae: ", test_mae)

test_loss, test_mae, test_mse = model_pose.evaluate(x_test, y_test[:,:,:,:2], verbose=0)
print("pose  train mae: ", test_mae)
#
#print("answer")
#print(y_test[0])
#print("predict")
#print(model.predict(x_test[:1]))


if "v" in args[3]:
    cv2.imshow('test',x_test[0] * 1.0 + 0.5)
    cv2.waitKey(1)

    hoge = np.ones((1, 25, 25, 1)) * 0.1
    print(model_pose.predict(hoge))
    print(model_height.predict(hoge))
    hoge[0, 3:6, :, 0] = np.zeros((3, 25))
    print(model_pose.predict(hoge))
    print(model_height.predict(hoge))
    hoge[0, :, 2:7, 0] = np.zeros((25, 5))
    print(model_pose.predict(hoge))
    print(model_height.predict(hoge))

    for i in range(1):
        print("answer")
        print(y_train[i])
        print("predict")
        print(model_pose.predict(x_train[i:i+1]))
        print(model_height.predict(x_train[i:i+1]))
        cv2.imshow('test',x_train[i] * 1.0 + 0.5)
        cv2.waitKey(1)
        #time.sleep(5)
        print("---")

    for i in range(5):
        print("answer")
        print(y_test[i])
        print("predict")
        print(model_pose.predict(x_test[i:i+1]))
        print(model_height.predict(x_test[i:i+1]))
        cv2.imshow('test',x_test[i] * 1.0 + 0.5)
        cv2.waitKey(1)
        #time.sleep(5)
        print("---")
