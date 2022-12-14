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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
            #while (terrains_y[terrain_index][0] != 1):
            while (terrains_y[terrain_index][0] == -1):#着地不可能領域を含める
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
        #surface_index = math.floor(random.random()*len(surfaces))
        #surface_x = math.floor(random.random() * (surfaces[surface_index].shape[1] - 25))
        #surface_y = math.floor(random.random() * (surfaces[surface_index].shape[0] - 25))
        #x_data[i, 14:39, 14:39, 0] += surfaces[surface_index][surface_y:surface_y+25, surface_x:surface_x+25]
        
        #tilt
        p = 2.0*random.random() - 1.0
        r = 2.0*random.random() - 1.0
        s = 1.0*random.random() - 0.5
        p = 0
        r = 0
        s = 0
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

model_pose = cnn_models.cnn_pose((H,W,C), "../checkpoints/checkpoint")
model_pose.summary()
model_height = cnn_models.cnn_height((H,W,C), "../checkpoints/checkpoint")
model_height.summary()

cnn_height_diff = 0
pca_height_diff = 0
cnn_pose_diff = 0
pca_pose_diff = 0

for i in range(x_train.shape[0]):
    print(i)
    cnn_pose_y = model_pose.predict(x_train[i:i+1])
    cnn_height_y = model_height.predict(x_train[i:i+1])

    data = x_train[i]
    average = np.average(data)
    cloud = np.zeros((data.shape[0] * data.shape[1], 3))
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            cloud[y*data.shape[1]+x, 0] = 0.01*x
            cloud[y*data.shape[1]+x, 1] = 0.01*y
            cloud[y*data.shape[1]+x, 2] = data[y, x, 0]
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(cloud, np.empty((0)))
    n = eigenvectors[2]
    if n[2] < 0:
        n = -n
    
    cnn_height_diff += np.abs(cnn_height_y[0,0,0,0] - y_train[i,0,0,2])
    pca_height_diff += np.abs(average - y_train[i,0,0,2])

    answer_x = np.array([1.0, 0.0, y_train[i,0,0,0]])
    answer_y = np.array([0.0, 1.0, y_train[i,0,0,1]])
    answer_n = np.cross(answer_x,answer_y)
    answer_n = answer_n / np.linalg.norm(answer_n)
    cnn_x = np.array([1.0, 0.0, cnn_pose_y[0,0,0,0]])
    cnn_y = np.array([0.0, 1.0, cnn_pose_y[0,0,0,1]])
    cnn_n = np.cross(cnn_x,cnn_y)
    cnn_n = cnn_n / np.linalg.norm(cnn_n)

    cnn_pose_diff += answer_n.dot(cnn_n)
    pca_pose_diff += answer_n.dot(n)

    #print(-n[0]/n[2], -n[1]/n[2], average)
    #print(y_train[i])
    #print(cnn_pose_y, cnn_height_y)

    #answercloud = np.zeros((data.shape[0] * data.shape[1], 3))
    #cnncloud = np.zeros((data.shape[0] * data.shape[1], 3))
    #pcacloud = np.zeros((data.shape[0] * data.shape[1], 3))
    #for y in range(data.shape[0]):
    #    for x in range(data.shape[1]):
    #        answercloud[y*data.shape[1]+x, 0] = 0.01*x
    #        answercloud[y*data.shape[1]+x, 1] = 0.01*y
    #        answercloud[y*data.shape[1]+x, 2] = 0.01*((x-12)*y_train[i,0,0,0] + (y-12)*y_train[i,0,0,1])+y_train[i,0,0,2]
    #        pcacloud[y*data.shape[1]+x, 0] = 0.01*x
    #        pcacloud[y*data.shape[1]+x, 1] = 0.01*y
    #        pcacloud[y*data.shape[1]+x, 2] = 0.01*((x-12)*(-n[0]/n[2]) + (y-12)*(-n[1]/n[2]))+average
    #        cnncloud[y*data.shape[1]+x, 0] = 0.01*x
    #        cnncloud[y*data.shape[1]+x, 1] = 0.01*y
    #        cnncloud[y*data.shape[1]+x, 2] = 0.01*((x-12)*cnn_pose_y[0,0,0,0] + (y-12)*cnn_pose_y[0,0,0,1])+cnn_height_y[0,0,0,0]
    #
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2] - average, label="data")
    #ax.scatter(answercloud[:,0], answercloud[:,1], answercloud[:,2] - average, label="correct")
    #ax.scatter(cnncloud[:,0], cnncloud[:,1], cnncloud[:,2] - average, label="cnn")
    #ax.scatter(pcacloud[:,0], pcacloud[:,1], pcacloud[:,2] - average, label="pca")
    #ax.set_zlim(-0.1, 0.1)
    #ax.legend(bbox_to_anchor=(0, 0), loc='upper left')
    #plt.show()

print(cnn_height_diff/x_train.shape[0], pca_height_diff/x_train.shape[0])
print((cnn_pose_diff)/x_train.shape[0], (pca_pose_diff)/x_train.shape[0])
