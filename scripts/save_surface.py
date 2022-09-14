#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import random
import math
import cv2
import time
import sys
import os
import rospy
import roslib
import ros_numpy
import datetime
from sensor_msgs.msg import Image
#import cnn_models

args = sys.argv
if len(args) == 1:
    args.append("")
print(args)

class SurfaceSaver:
    def __init__(self):
        self.drawing = False
        self.mouse_x1 = -1
        self.mouse_y1 = -1
        self.mouse_x2 = -1
        self.mouse_y2 = -1
        self.color = 0
        self.sum_image = np.zeros((47, 47))
        self.sum_counter = 0
        rospy.Subscriber('rt_filtered_current_heightmap/output', Image, self.surfaceSaver, queue_size=1, buff_size=2**24)

    def mousePoints(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x1 = self.mouse_x2 = math.floor(x/2.0)
            self.mouse_y1 = self.mouse_y2 = math.floor(y/2.0)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.mouse_x2 = math.floor(x/2.0)
                self.mouse_y2 = math.floor(y/2.0)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_x2 = math.floor(x/2.0)
            self.mouse_y2 = math.floor(y/2.0)
            self.drawing = False

    def surfaceSaver(self, msg):
        input_image = ros_numpy.numpify(msg)
        input_image = input_image[:, :, 0]
        input_image = input_image - (input_image<-1e5)*input_image + (input_image<-1e5) * cv2.blur(input_image - (input_image<-1e5)*input_image, (9, 9)) / cv2.blur((input_image>-1e5).astype(float), (9, 9))
        np.nan_to_num(input_image, copy=False, nan=-1e10)
        vision_image = input_image.copy()
        vision_image = np.where(vision_image > 0.5, 0.5, vision_image)
        vision_image = np.where(vision_image < -0.5, -0.5, vision_image)
        vision_image = vision_image + 0.5
        while True:
            tmp_image = vision_image.copy()
            self.color += 0.014
            if self.color > 2:
                self.color -= 2
            if self.mouse_x2 < 0 or self.mouse_y2 < 0: #未選択
                pass
            elif self.mouse_x1 == self.mouse_x2 and self.mouse_y1 == self.mouse_y2: #terrain
                #cv2.circle(tmp_image, (self.mouse_x1, self.mouse_y1), 2, math.fabs(self.color-1), thickness=-1)
                cv2.rectangle(tmp_image, (self.mouse_x1-1, self.mouse_y1-1), (self.mouse_x2+1, self.mouse_y2+1), math.fabs(self.color - 1), 1)
                cv2.rectangle(tmp_image, (self.mouse_x1-11, self.mouse_y1-11), (self.mouse_x2+11, self.mouse_y2+11), math.fabs(self.color - 1), 1)
                cv2.rectangle(tmp_image, (self.mouse_x1-20, self.mouse_y1-20), (self.mouse_x2+20, self.mouse_y2+20), math.fabs(self.color - 1), 1)
                cv2.rectangle(tmp_image, (self.mouse_x1-23, self.mouse_y1-23), (self.mouse_x2+23, self.mouse_y2+23), math.fabs(self.color - 1), 1)
            else: #surface
                cv2.rectangle(tmp_image, (self.mouse_x1, self.mouse_y1), (self.mouse_x2, self.mouse_y2), math.fabs(self.color - 1), 2)
            tmp_image = cv2.resize(tmp_image, (tmp_image.shape[1]*2, tmp_image.shape[0]*2))
            cv2.imshow("surface_saver", tmp_image)
            cv2.setMouseCallback("surface_saver", self.mousePoints)

            key = cv2.waitKey(10)
            if key == 46 or key == 8 or key == 255: #delete, backspace
                self.mouse_x1 = self.mouse_x2 = self.mouse_y1 = self.mouse_y2 = -1
                print("cancel")
                break

            if self.mouse_x2 < 0 or self.mouse_y2 < 0: #未選択
                pass
            elif self.mouse_x1 == self.mouse_x2 and self.mouse_y1 == self.mouse_y2: #terrain
                save_image = input_image[self.mouse_y1-23:self.mouse_y2+24, self.mouse_x1-23:self.mouse_x2+24]
                if key == 111: #O
                    nowdate = datetime.datetime.now()
                    np.savetxt("../terrains/x/"+args[1]+nowdate.strftime('%y%m%d_%H%M%S')+".csv", save_image, delimiter=",")
                    np.savetxt("../terrains/y/"+args[1]+nowdate.strftime('%y%m%d_%H%M%S')+".csv", np.ones((3, 3)), delimiter=",")
                    self.mouse_x1 = self.mouse_x2 = self.mouse_y1 = self.mouse_y2 = -1
                    print("save steppable image")
                    break
                elif key == 120: #X
                    nowdate = datetime.datetime.now()
                    np.savetxt("../terrains/x/"+args[1]+nowdate.strftime('%y%m%d_%H%M%S')+".csv", save_image, delimiter=",")
                    np.savetxt("../terrains/y/"+args[1]+nowdate.strftime('%y%m%d_%H%M%S')+".csv", np.zeros((3, 3)), delimiter=",")
                    self.mouse_x1 = self.mouse_x2 = self.mouse_y1 = self.mouse_y2 = -1
                    print("save steppable image")
                    break
            else: #surfece
                if key == 13: #Enter
                    if self.mouse_x1 > self.mouse_x2:
                        tmp = self.mouse_x1
                        self.mouse_x1 = self.mouse_x2
                        self.mouse_x2 = tmp
                    if self.mouse_y1 > self.mouse_y2:
                        tmp = self.mouse_y1
                        self.mouse_y1 = self.mouse_y2
                        self.mouse_y2 = tmp
                    if self.mouse_x2 - self.mouse_x1 < 47 or self.mouse_y2 - self.mouse_y1 < 47:
                        print("size error")
                        self.mouse_x1 = self.mouse_x2 = self.mouse_y1 = self.mouse_y2 = -1
                        break
                    save_image = input_image[self.mouse_y1 : self.mouse_y2, self.mouse_x1 : self.mouse_x2]
                    np.savetxt("../surfaces/"+args[1]+datetime.datetime.now().strftime('%y%m%d_%H%M%S')+".csv", save_image, delimiter=",")
                    self.mouse_x1 = self.mouse_x2 = self.mouse_y1 = self.mouse_y2 = -1
                    print("save surface image")
                    break

if __name__ == "__main__":
    rospy.init_node('surface_saver')
    surface_saver = SurfaceSaver()
    rospy.spin()
