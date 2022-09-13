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

#----------------------------
# データの作成

terrains_x = []
terrains_y = []
surfaces = []
for filename in os.listdir("../terrains/x"):
    terrains_x.append(np.loadtxt("../terrains/x/"+filename, delimiter=","))
    terrains_y.append(np.loadtxt("../terrains/y/"+filename, delimiter=","))
for filename in os.listdir("../surfaces"):
    surfaces.append(np.loadtxt("../surfaces/"+filename, delimiter=","))

pitch = np.zeros((47, 47, 1))
roll  = np.zeros((47, 47, 1))
scale = np.ones ((47, 47, 1))

for i in range(47):
    for j in range(47):
        pitch[i, j, 0] = (j-23)*0.01
        roll[i, j, 0]  = (i-23)*0.01

def randomize(x, y):
    randomizer = np.arange(x.shape[0])
    np.random.shuffle(randomizer)
    return x[randomizer], y[randomizer]

def generate_data(num):
    global pitch, roll, scale, steppable_terrains, unsteppable_terrains, surfaces
    x_data = np.zeros((num, 47, 47, 1))
    y_data = np.zeros((num, 5, 5, 1))

    for i in range(num):
        #surface
        surface_index = math.floor(random.random()*len(surfaces))
        surface_x = math.floor(random.random() * (surfaces[surface_index].shape[1] - 47))
        surface_y = math.floor(random.random() * (surfaces[surface_index].shape[0] - 47))
        x_data[i, :, :, 0] += surfaces[surface_index][surface_y:surface_y+47, surface_x:surface_x+47]

        #rotate
        theta = random.random() * 360
        M = cv2.getRotationMatrix2D((math.floor(x_data.shape[2] / 2.), math.floor(x_data.shape[1] / 2.)), theta, 1)
        x_data[i, :, :, 0] = cv2.warpAffine(x_data[i], M, (x_data.shape[2], x_data.shape[1]))

        #terrain
        terrain_index = math.floor(random.random()*len(terrains_x))
        x_data[i, :, :, 0] += terrains_x[terrain_index]
        y_data[i, :, :, 0] += terrains_y[terrain_index]

    #for i in range(math.floor(num*0.7)):
    #    begin, end = np.sort([math.floor(random.random()*(48)), math.floor(random.random()*(48))])

    #    l = end - begin
    #    if l == 0 :
    #        continue
    #    h = random.random() * 0.6 - 0.3
    #    x_data[i, begin : end, :, 0] += np.ones((l, x_data.shape[2])) * h
    #    if h > 0.02:
    #        if begin > 13 or end < 33:
    #            y_data[i] = np.zeros((3, 3, 1))
    #    elif h < -0.02:
    #        if not (end <= 13 or begin >= 34 or (begin >= 17 and end <=30)):
    #            y_data[i] = np.zeros((3, 3, 1))
    #for i in range(num):
    #    theta = random.random() * 60 + 60
    #    M = cv2.getRotationMatrix2D((math.floor(x_data.shape[2] / 2.), math.floor(x_data.shape[1] / 2.)), theta, 1)
    #    x_data[i, :, :, 0] = cv2.warpAffine(x_data[i], M, (x_data.shape[2], x_data.shape[1]))
    #x_data, y_data = randomize(x_data, y_data)

    #for i in range(math.floor(num*0.7)):
    #    begin, end = np.sort([math.floor(random.random()*(48)), math.floor(random.random()*(48))])

    #    l = end - begin
    #    if l == 0 :
    #        continue
    #    h = random.random() * 0.6 - 0.3
    #    x_data[i, begin : end, :, 0] += np.ones((l, x_data.shape[2])) * h
    #    if h > 0.02:
    #        if begin > 13 or end < 33:
    #            y_data[i] = np.zeros((3, 3, 1))
    #    elif h < -0.02:
    #        if not (end <= 13 or begin >= 34 or (begin >= 17 and end <=30)):
    #            y_data[i] = np.zeros((3, 3, 1))
    #x_data, y_data = randomize(x_data, y_data)

    #for i in range(num):

        #rotate
        theta = random.random() * 360
        M = cv2.getRotationMatrix2D((math.floor(x_data.shape[2] / 2.), math.floor(x_data.shape[1] / 2.)), theta, 1)
        x_data[i, :, :, 0] = cv2.warpAffine(x_data[i], M, (x_data.shape[2], x_data.shape[1]))
        M = cv2.getRotationMatrix2D((math.floor(y_data.shape[2] / 2.), math.floor(y_data.shape[1] / 2.)), theta, 1)
        y_data[i, :, :, 0] = cv2.warpAffine(y_data[i], M, (y_data.shape[2], y_data.shape[1]))

        #tilt
        p = 2.0*random.random() - 1.0
        r = 2.0*random.random() - 1.0
        s = 1.0*random.random() - 0.5
        x_data[i] += p * pitch + r * roll + s * scale + 0.05*np.random.rand(47, 47, 1) - 0.025 * np.ones((47, 47, 1))
    x_data, y_data = randomize(x_data, y_data)

    x_data = x_data[:, 3:44, 3:44]
    y_data = y_data[:, 1:4, 1:4]
    y_data = y_data > 0.5
    y_data = y_data.astype(np.int)
    return x_data, y_data

x_train, y_train = generate_data(int(args[1]))
x_test, y_test = generate_data(100)

#----------------------------
# 学習
if "w" in args[3]:
    model_steppable_region = cnn_models.cnn_steppable((41,41,1), "../checkpoints/checkpoint")
else:
    model_steppable_region = cnn_models.cnn_steppable((41,41,1), "")
if "v" in args[3]:
    model_steppable_region.summary()
if "f" in args[3]:
    print(x_train.shape)
    print(y_train.shape)
    model_steppable_region.fit(x_train, y_train, batch_size=100, epochs=int(args[2]))
    model_steppable_region.save_weights('../checkpoints/checkpoint/checkpoint_steppable_region')

#---------------------------

#----------------------------
# 学習データに対する評価
train_loss, train_accuracy = model_steppable_region.evaluate(x_train, y_train, verbose=0)
#train_loss, train_mae, train_mse = model_height.evaluate(x_train, y_train[:,:,:,2], verbose=2)
print('Train data loss:', train_loss)
print('Train data accuracy:', train_accuracy)
#print("Testing set Mean Abs Error: {:5.2f} MPG".format(train_mae))

#----------------------------

#----------------------------
# 評価データに対する評価
test_loss, test_accuracy = model_steppable_region.evaluate(x_test, y_test, verbose=0)
print('Test data loss:', test_loss)
print('Test data accuracy:', test_accuracy)
#test_loss, test_mae, test_mse = model.evaluate(x_test, y_test, verbose=2)
#print('Test data loss:', test_loss)
#print("Testing set Mean Abs Error: {:5.2f} MPG".format(test_mae))


if "v" in args[3]:
    cv2.imshow('test',x_test[0] * 1.0 + 0.5)

    for i in range(5):
        print("answer")
        print(y_train[i])
        print("predict")
        print(np.argmax(model_steppable_region.predict(x_train[i:i+1]), axis=3))
        cv2.imshow('test',x_train[i]*2.0 + 0.5)
        cv2.waitKey(1)
        time.sleep(2)
        print("---")

    print("answer")
    print(y_test[70])
    print("predict")
    print(np.argmax(model_steppable_region.predict(x_test[70:71]), axis=3))
    cv2.imshow('test',x_test[70] * 2.0 + 0.5)
    cv2.waitKey(1)
    time.sleep(5)
    print("---")
