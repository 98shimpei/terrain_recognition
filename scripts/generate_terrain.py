#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import random
import math
import cv2
import time
import datetime
import sys
import cnn_models
import os

args = sys.argv
if len(args) == 1:
    args.append(10)
if len(args) == 2:
    args.append("test")
print(args)

nowdate = datetime.datetime.now()
for n in range(int(args[1])):
    x_data = np.zeros((47, 47))
    y_data = np.zeros((5, 5))
    for i in range(math.floor(random.random() * 2)):
        if random.random() < 0.7: #長方形溝
            begin, end = np.sort([math.floor(random.random()*(48)), math.floor(random.random()*(48))])
            if begin == end:
                continue
            h = random.random() * 0.6 - 0.3
            x_data[begin:end, :] += h
        else: #台形溝
            begin, mid1, mid2, end = np.sort([math.floor(random.random()*48), math.floor(random.random()*48), math.floor(random.random()*48), math.floor(random.random()*48)])
            if begin == end:
                continue
            h = random.random() * 0.6 - 0.3
            for i in range(begin, mid1):
                x_data[i, :] += h*(i-begin+1)/(mid1-begin+1)
            x_data[mid1:mid2, :] += h
            for i in range(mid2, end):
                x_data[i, :] += h*(i-end)/(mid2-1-end)
        theta = random.random() * 360
        M = cv2.getRotationMatrix2D((math.floor(x_data.shape[1] / 2.), math.floor(x_data.shape[0] / 2.)), theta, 1)
        x_data = cv2.warpAffine(x_data, M, (x_data.shape[1], x_data.shape[0]))
    for i in range(math.floor(random.random() * 7)): #穴
        l = math.floor(random.random()*3+3)
        x = math.floor(random.random()*(48-l))
        y = math.floor(random.random()*(48-l))
        h = random.random() * 0.3
        x_data[x:x+l, y:y+l] -= h
    max_h = np.max(x_data[13:34, 13:34])
    x_data -= max_h
    median = cv2.medianBlur(x_data.astype(np.float32), 5)
    if np.max(x_data) > 0.03:
        y_data = np.zeros((5, 5))
    for y in range(-2, 3):
        for x in range(-2, 3):
            if (((np.max(median[14+y,14+x:24+x]))>-0.02 and np.max(median[14+y,23+x:33+x])>-0.02) and
                ((np.max(median[32+y,14+x:24+x]))>-0.02 and np.max(median[32+y,23+x:33+x])>-0.02) and
                ((np.max(median[14+y:24+y,14+x]))>-0.02 and np.max(median[23+y:33+y,14+x])>-0.02) and
                ((np.max(median[14+y:24+y,32+x]))>-0.02 and np.max(median[23+y:33+y,32+x])>-0.02)):
                y_data[1+y,1+x]=1
            else:
                y_data[1+y,1+x]=0
    np.savetxt("../terrains/x/"+args[1]+nowdate.strftime('%y%m%d_%H%M%S')+"_"+str(n)+".csv", x_data, delimiter=",")
    np.savetxt("../terrains/y/"+args[1]+nowdate.strftime('%y%m%d_%H%M%S')+"_"+str(n)+".csv", y_data, delimiter=",")
