#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tef
import numpy as np
import random
import math
import time
import cv2
import rospy
import roslib
import quaternion
import copy
import p2t
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import HeightmapConfig
from jsk_recognition_msgs.msg import PolygonArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import Point32
from geometry_msgs.msg import PolygonStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2

class DelayChecker:
    def __init__(self):
        self.times = {}
        rospy.Subscriber('rs_l515/depth_registered/points', PointCloud2, self.rawPointCloudCallback, queue_size=1)
        rospy.Subscriber('rs_l515/depth/color/points_low_hz', PointCloud2, self.lowHzPointCloudCallback, queue_size=1)
        rospy.Subscriber('rs_l515/depth/color/points_low_hz_resized', PointCloud2, self.lowHzResizedPointCloudCallback, queue_size=1)
        rospy.Subscriber('rs_l515/depth/color/points_filtered', PointCloud2, self.filteredPointCloudCallback, queue_size=1)
        rospy.Subscriber('self_filtered_pointcloud/output', PointCloud2, self.centerPointCloudCallback, queue_size=1)
        rospy.Subscriber('object_bbox_clipper/output', PointCloud2, self.objectPointCloudCallback, queue_size=1)
        rospy.Subscriber('rt_current_heightmap/output', Image, self.currentHeightmapCallback, queue_size=1)
        rospy.Subscriber('rt_filtered_current_heightmap/output', Image, self.filteredHeightmapCallback, queue_size=1)
        rospy.Subscriber('trimmed_image_output', Image, self.trimmedImageCallback, queue_size=1)

    def rawPointCloudCallback(self, msg):
        self.times[msg.header.stamp] = []
        self.times[msg.header.stamp].append(["stamp", msg.header.stamp])
        self.times[msg.header.stamp].append(["raw", rospy.Time.now()])

    def lowHzPointCloudCallback(self, msg):
        self.times[msg.header.stamp].append(["low_hz", rospy.Time.now()])

    def lowHzResizedPointCloudCallback(self, msg):
        self.times[msg.header.stamp].append(["resized", rospy.Time.now()])

    def filteredPointCloudCallback(self, msg):
        self.times[msg.header.stamp].append(["filtered pc", rospy.Time.now()])

    def centerPointCloudCallback(self, msg): 
        self.times[msg.header.stamp].append(["center", rospy.Time.now()])

    def objectPointCloudCallback(self, msg): 
        self.times[msg.header.stamp].append(["bbox", rospy.Time.now()])

    def currentHeightmapCallback(self, msg): 
        self.times[msg.header.stamp].append(["current", rospy.Time.now()])

    def filteredHeightmapCallback(self, msg):
        self.times[msg.header.stamp].append(["filtered hm", rospy.Time.now()])

    def trimmedImageCallback(self, msg):
        self.times[msg.header.stamp].append(["trimmed", rospy.Time.now()])
        for i in range(1, len(self.times[msg.header.stamp])):
            tmp = self.times[msg.header.stamp][i][1] - self.times[msg.header.stamp][i-1][1]
            tmp_zero = self.times[msg.header.stamp][i][1] - self.times[msg.header.stamp][0][1]
            print("{} to {} : {} s {} ms : {} s {} ms".format(self.times[msg.header.stamp][i-1][0], self.times[msg.header.stamp][i][0], tmp.secs, math.floor(tmp.nsecs/1000000), tmp_zero.secs, math.floor(tmp_zero.nsecs/1000000)))
        tmp = self.times[msg.header.stamp][-1][1] - self.times[msg.header.stamp][0][1]
        print("{} to {} : {} s {} ms".format(self.times[msg.header.stamp][-1][0], self.times[msg.header.stamp][0][0], tmp.secs, math.floor(tmp.nsecs/1000000)))

        print("")

if __name__=='__main__':
    rospy.init_node('delay_checker')
    delay_checker = DelayChecker()
    rospy.spin()

