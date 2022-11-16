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
from scipy.spatial import ConvexHull

def is_inner_polygon(array, indices, points):
    if array[indices[0]][2] < -50 or array[indices[1]][2] < -50 or array[indices[2]][2] < -50:
        return False
    c1 = (array[indices[1]] - array[indices[0]])[0] * (points - array[indices[1]])[1] - (array[indices[1]] - array[indices[0]])[1] * (points - array[indices[1]])[0]
    c2 = (array[indices[2]] - array[indices[1]])[0] * (points - array[indices[2]])[1] - (array[indices[2]] - array[indices[1]])[1] * (points - array[indices[2]])[0]
    c3 = (array[indices[0]] - array[indices[2]])[0] * (points - array[indices[0]])[1] - (array[indices[0]] - array[indices[2]])[1] * (points - array[indices[0]])[0]
    return (c1>=0 and c2>=0 and c3>=0) or (c1<=0 and c2<=0 and c3<=0)

args = sys.argv
if len(args) == 1:
    args.append(10)
if len(args) == 2:
    args.append("generate")
print(args)

nowdate = datetime.datetime.now()
for num in range(int(args[1])):
    x_data = np.zeros((47, 47))
    y_data = np.array([1])
    for i in range(math.floor(random.random() * 1.7 + 0.9)):
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
    x_data = cv2.medianBlur(x_data.astype(np.float32), 5)


    center_data = x_data[13:34, 13:34]
    array = []
    for y in range(center_data.shape[0]):
        for x in range(center_data.shape[1]):
            array.append([x, y, center_data[y, x]])
    array.append([0,0,-100])
    array.append([0,20,-100])
    array.append([20,0,-100])
    array.append([20,20,-100])
    array = np.array(array)
    hull = ConvexHull(array)
    n = np.array([0,0,0])
    a = np.array([0,0,0])
    for simplice in hull.simplices:
        if is_inner_polygon(array, simplice, np.array([10, 10, center_data[10,10]])):
            n = np.cross(array[simplice[1]] - array[simplice[0]], array[simplice[2]] - array[simplice[1]])
            n = n / np.linalg.norm(n) * np.sign(n[2])
            a = array[simplice[0]]
            break
    contact_region = np.zeros((center_data.shape[1], center_data.shape[0]), dtype=np.uint8)
    for y in range(center_data.shape[0]):
        for x in range(center_data.shape[1]):
            distance = np.dot(np.array([y, x, center_data[y, x]]) - a, n)
            if distance > -0.01:
                contact_region[y,x] = 255
    contours, hierarchy = cv2.findContours(contact_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tmp_array = np.empty((0, 2), dtype=np.int32)
    for cnt in contours:
        if cv2.contourArea(cnt) > 8:
            tmp_array = np.concatenate([tmp_array, cnt.reshape([int(cnt.size/2),2])])
    if tmp_array.size == 0:
        y_data = np.array([0])
    else:
        cvhull = cv2.convexHull(tmp_array)
        if not(cv2.pointPolygonTest(cvhull, (0, 10), False) >= 0 and cv2.pointPolygonTest(cvhull, (10, 0), False) >= 0 and cv2.pointPolygonTest(cvhull, (20, 10), False) >= 0 and cv2.pointPolygonTest(cvhull, (10, 20), False) >= 0):
            y_data = np.array([0])
        else:
            for y in range(x_data.shape[0]):
                for x in range(x_data.shape[1]):
                    distance = np.dot(np.array([y, x, x_data[y, x]]) - a, n)
                    if distance > 0.03:
                        y_data = np.array([0])


    np.savetxt("../terrains/x/"+args[2]+nowdate.strftime('%y%m%d_%H%M%S')+"_"+str(num)+".csv", x_data, delimiter=",")
    np.savetxt("../terrains/y/"+args[2]+nowdate.strftime('%y%m%d_%H%M%S')+"_"+str(num)+".csv", y_data, delimiter=",")
