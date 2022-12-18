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
import copy
from sensor_msgs.msg import Image
import cnn_models
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

args = sys.argv
if len(args) == 1:
    args.append("")
print(args)

class SurfaceSaver:
    def __init__(self):
        self.drag = False
        self.calc_pose = False
        self.mouse_x1 = -1
        self.mouse_y1 = -1
        self.mouse_x2 = -1
        self.mouse_y2 = -1
        self.color = 0
        self.sum_image = np.zeros((53, 53))
        self.sum_counter = 0
        self.model_steppable = cnn_models.cnn_steppable((37, 37, 1), "../checkpoints/checkpoint")
        self.model_height = cnn_models.cnn_height((21, 21, 1), "../checkpoints/checkpoint")
        self.model_pose = cnn_models.cnn_pose((21, 21, 1), "../checkpoints/checkpoint")

        self.marker_publisher = rospy.Publisher('landing_pose_marker', Marker, queue_size = 1)
        rospy.Subscriber('rt_filtered_current_heightmap/output', Image, self.surfaceSaver, queue_size=1, buff_size=2**24)

    def mousePoints(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x1 = self.mouse_x2 = math.floor(x/2.0)
            self.mouse_y1 = self.mouse_y2 = math.floor(500-y/2.0)
            self.drag = True
            self.calc_pose = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag:
                self.mouse_x2 = math.floor(x/2.0)
                self.mouse_y2 = math.floor(500-y/2.0)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_x2 = math.floor(x/2.0)
            self.mouse_y2 = math.floor(500-y/2.0)
            self.drag = False

    def publishPoseMsg(self, header, action, start_pos, ez, is_steppable):
        end_pos = start_pos + 0.3 * ez
        pose_msg = Marker()
        pose_msg.header = copy.deepcopy(header)
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.ns = "recognized_pose"
        pose_msg.id = 0
        pose_msg.type = Marker.ARROW
        pose_msg.action = action
        start = Point()
        start.x = start_pos[0]
        start.y = start_pos[1]
        start.z = start_pos[2]
        end = Point()
        end.x = end_pos[0]
        end.y = end_pos[1]
        end.z = end_pos[2]
        pose_msg.points.append(start)
        pose_msg.points.append(end)
        if is_steppable:
            pose_msg.color.r = 0.2
            pose_msg.color.g = 1.0
            pose_msg.color.b = 0.2
        else:
            pose_msg.color.r = 1.0
            pose_msg.color.g = 0.2
            pose_msg.color.b = 0.2
        pose_msg.color.a = 1.0
        pose_msg.scale.x = 0.03
        pose_msg.scale.y = 0.05
        pose_msg.scale.z = 0.07
        pose_msg.pose.orientation.w = 1.0
        self.marker_publisher.publish(pose_msg)

    def surfaceSaver(self, msg):
        input_image = ros_numpy.numpify(msg)
        input_image = input_image[:, :, 0]
        input_image = input_image - (input_image<-1e5)*input_image + (input_image<-1e5) * cv2.blur(input_image - (input_image<-1e5)*input_image, (9, 9)) / cv2.blur((input_image>-1e5).astype(float), (9, 9))
        np.nan_to_num(input_image, copy=False, nan=-1e10)
        vision_image = input_image.copy() * 1.2
        vision_image = np.where(vision_image > 0.5, 0.5, vision_image)
        vision_image = np.where(vision_image < -0.5, -0.5, vision_image)
        vision_image = vision_image + 0.5
        while True:
            if self.calc_pose:
                self.calc_pose = False
                predict_image = input_image.copy()
                predict_image = predict_image.reshape((1,input_image.shape[0],input_image.shape[1],1))
                is_steppable = np.argmax(self.model_steppable.predict(predict_image[:,self.mouse_y1-18:self.mouse_y1+19, self.mouse_x1-18:self.mouse_x1+19, :]))
                height = self.model_height.predict(predict_image[:, self.mouse_y1-10:self.mouse_y1+11, self.mouse_x1-10:self.mouse_x1+11, :])[0,0,0,0]
                pose = self.model_pose.predict(predict_image[:, self.mouse_y1-10:self.mouse_y1+11, self.mouse_x1-10:self.mouse_x1+11, :])[0,0,0]
                tmp_vecx = [1.0, 0.0, pose[0]]
                tmp_vecy = [0.0, 1.0, pose[1]]
                tmp_vecz = np.cross(tmp_vecx, tmp_vecy)
                tmp_vecz = tmp_vecz / np.linalg.norm(tmp_vecz)
                print("height: ", height, " pose: ", pose)
            tmp_image = vision_image.copy()
            self.color += 0.014
            if self.color > 2:
                self.color -= 2
            if self.mouse_x2 < 0 or self.mouse_y2 < 0: #未選択
                pass
            elif self.mouse_x1 == self.mouse_x2 and self.mouse_y1 == self.mouse_y2: #terrain
                #cv2.circle(tmp_image, (self.mouse_x1, self.mouse_y1), 2, math.fabs(self.color-1), thickness=-1)
                cv2.rectangle(tmp_image, (self.mouse_x1-1, self.mouse_y1-1), (self.mouse_x2+1, self.mouse_y2+1), math.fabs(self.color - 1), 1)
                cv2.rectangle(tmp_image, (self.mouse_x1-10, self.mouse_y1-10), (self.mouse_x2+10, self.mouse_y2+10), math.fabs(self.color - 1), 1)
                cv2.rectangle(tmp_image, (self.mouse_x1-18, self.mouse_y1-18), (self.mouse_x2+18, self.mouse_y2+18), math.fabs(self.color - 1), 1)
                cv2.rectangle(tmp_image, (self.mouse_x1-26, self.mouse_y1-26), (self.mouse_x2+26, self.mouse_y2+26), math.fabs(self.color - 1), 1)
            else: #surface
                cv2.rectangle(tmp_image, (self.mouse_x1, self.mouse_y1), (self.mouse_x2, self.mouse_y2), math.fabs(self.color - 1), 2)
            tmp_image = cv2.resize(tmp_image, (tmp_image.shape[1]*2, tmp_image.shape[0]*2))
            tmp_image = cv2.flip(tmp_image, 0)
            cv2.imshow("surface_saver", tmp_image)
            cv2.setMouseCallback("surface_saver", self.mousePoints)

            key = cv2.waitKey(10)
            if key == 46 or key == 8 or key == 255: #delete, backspace
                self.mouse_x1 = self.mouse_x2 = self.mouse_y1 = self.mouse_y2 = -1
                if not self.calc_pose:
                    self.publishPoseMsg(msg.header, Marker.DELETE, np.array([(self.mouse_x1)*0.01, (self.mouse_y1-250)*0.01, height]), tmp_vecz, is_steppable)
                print("cancel")
                break

            if self.mouse_x2 < 0 or self.mouse_y2 < 0: #未選択
                pass
            elif self.mouse_x1 == self.mouse_x2 and self.mouse_y1 == self.mouse_y2: #terrain
                save_image = input_image[self.mouse_y1-23:self.mouse_y2+24, self.mouse_x1-23:self.mouse_x2+24]
                if not self.calc_pose:
                    self.publishPoseMsg(msg.header, Marker.ADD, np.array([(self.mouse_x1)*0.01, (self.mouse_y1-250)*0.01, height]), tmp_vecz, is_steppable)

                if key == 111: #O
                    nowdate = datetime.datetime.now()
                    np.savetxt("../terrains/x/"+args[1]+nowdate.strftime('%y%m%d_%H%M%S')+".csv", save_image, delimiter=",")
                    np.savetxt("../terrains/y/"+args[1]+nowdate.strftime('%y%m%d_%H%M%S')+".csv", np.array([1.0, 0.0, 0.0, 1.0, 0.0]), delimiter=",")
                    self.mouse_x1 = self.mouse_x2 = self.mouse_y1 = self.mouse_y2 = -1
                    print("save steppable image")
                    break
                elif key == 120: #X
                    nowdate = datetime.datetime.now()
                    np.savetxt("../terrains/x/"+args[1]+nowdate.strftime('%y%m%d_%H%M%S')+".csv", save_image, delimiter=",")
                    np.savetxt("../terrains/y/"+args[1]+nowdate.strftime('%y%m%d_%H%M%S')+".csv", np.array([-1.0, 0.0, 0.0, 1.0, 0.0]), delimiter=",")
                    self.mouse_x1 = self.mouse_x2 = self.mouse_y1 = self.mouse_y2 = -1
                    print("save steppable image")
                    break
                elif key == 112: #P
                    nowdate = datetime.datetime.now()
                    np.savetxt("../terrains/pose/"+args[1]+nowdate.strftime('%y%m%d_%H%M%S')+".csv", save_image, delimiter=",")
                    self.mouse_x1 = self.mouse_x2 = self.mouse_y1 = self.mouse_y2 = -1
                    print("save steppable image")
                    break
            else: #surfece
                if not self.calc_pose:
                    self.publishPoseMsg(msg.header, Marker.DELETE, np.array([(self.mouse_x1)*0.01, (self.mouse_y1-250)*0.01, height]), tmp_vecz, is_steppable)
                if key == 13: #Enter
                    if self.mouse_x1 > self.mouse_x2:
                        tmp = self.mouse_x1
                        self.mouse_x1 = self.mouse_x2
                        self.mouse_x2 = tmp
                    if self.mouse_y1 > self.mouse_y2:
                        tmp = self.mouse_y1
                        self.mouse_y1 = self.mouse_y2
                        self.mouse_y2 = tmp
                    if self.mouse_x2 - self.mouse_x1 < 53 or self.mouse_y2 - self.mouse_y1 < 53:
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
