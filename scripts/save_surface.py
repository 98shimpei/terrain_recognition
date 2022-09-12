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

class SurfaceSaver:
    def __init__(self):
        self.drawing = False
        self.mouse_x1 = -1
        self.mouse_y1 = -1
        self.mouse_x2 = -1
        self.mouse_y2 = -1
        self.color = 0
        rospy.Subscriber('rt_filtered_current_heightmap/output', Image, self.surfaceSaver, queue_size=1, buff_size=2**24)

    def mousePoints(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x1 = self.mouse_x2 = x
            self.mouse_y1 = self.mouse_y2 = y
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.mouse_x2 = x
                self.mouse_y2 = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_x2 = x
            self.mouse_y2 = y
            self.drawing = False

    def surfaceSaver(self, msg):
        input_image = ros_numpy.numpify(msg)
        input_image = input_image[:, :, 0]
        vision_image = input_image.copy()
        vision_image = np.where(vision_image > 0.5, 0.5, vision_image)
        vision_image = np.where(vision_image < -0.5, -0.5, vision_image)
        vision_image = vision_image + 0.5
        while True:
            tmp_image = vision_image.copy()
            self.color += 0.014
            if self.color > 2:
                self.color -= 2
            if self.mouse_x1 != self.mouse_x2:
                cv2.rectangle(tmp_image, (self.mouse_x1, self.mouse_y1), (self.mouse_x2, self.mouse_y2), math.fabs(self.color - 1), 2)
            cv2.imshow("surface_saver", tmp_image)
            cv2.setMouseCallback("surface_saver", self.mousePoints)
            key = cv2.waitKey(10)
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
                np.savetxt("../surfaces/"+datetime.datetime.now().strftime('%y%m%d_%H%M%S')+".csv", save_image, delimiter=",")
                self.mouse_x1 = self.mouse_x2 = self.mouse_y1 = self.mouse_y2 = -1
                print("save image")
                break
            elif key == 46 or key == 8 or key == 255: #delete, backspace
                self.mouse_x1 = self.mouse_x2 = self.mouse_y1 = self.mouse_y2 = -1
                print("cancel")
                break

if __name__ == "__main__":
    rospy.init_node('surface_saver')
    surface_saver = SurfaceSaver()
    rospy.spin()
