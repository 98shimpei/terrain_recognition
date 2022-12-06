#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tef
from tensorflow.keras import backend as K
import numpy as np
import random
import math
import time
import cv2
import rospy
import roslib
import tf
import quaternion
import copy
import p2t
import ros_numpy
import cnn_models
import threading
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import HeightmapConfig
from jsk_recognition_msgs.msg import PolygonArray
from terrain_recognition.msg import OnlineFootStep
from terrain_recognition.msg import SteppableRegion
from geometry_msgs.msg import Point
from geometry_msgs.msg import Point32
from geometry_msgs.msg import PolygonStamped
from visualization_msgs.msg import Marker

class SteppableRegionPublisher:
    def __init__(self):
        self.heightmap_config_flag = False
        self.fixed_frame = rospy.get_param("/accumulated_steppable_region_publisher/fixed_frame", "leg_odom")
        self.grid_length = rospy.get_param("/accumulated_steppable_region_publisher/grid_length", 1.0)
        self.update_pixel_erode = rospy.get_param("/accumulated_steppable_region_publisher/update_pixel_erode", 25)
        self.accumulate_length = rospy.get_param("/accumulated_steppable_region_publisher/accumulate_length", 500)
        self.accumulate_center_x = rospy.get_param("/accumulated_steppable_region_publisher/accumulate_center_x", 150)
        self.accumulate_center_y = rospy.get_param("/accumulated_steppable_region_publisher/accumulate_center_y", 250)
        self.trim_center_x = rospy.get_param("/accumulated_steppable_region_publisher/trim_center_x", 100)
        self.trim_center_y = rospy.get_param("/accumulated_steppable_region_publisher/trim_center_y", 100)
        self.trim_length = rospy.get_param("/accumulated_steppable_region_publisher/trim_length", 200)
        self.checkpoint_path = rospy.get_param("/accumulated_steppable_region_publisher/checkpoint_path", './checkpoint')
        self.debug_output = rospy.get_param("/accumulated_steppable_region_publisher/debug_output", True)

        self.listener = tf.TransformListener()
        self.center_H = np.identity(4)
        self.prev_H = np.identity(4)
        self.convex_list = []
        self.accumulated_steppable_image = np.zeros((self.accumulate_length, self.accumulate_length))
        self.accumulated_height_image = np.zeros((self.accumulate_length, self.accumulate_length))
        self.accumulated_pose_image = np.zeros((self.accumulate_length, self.accumulate_length, 2))
        self.accumulated_yaw_image = np.zeros((self.accumulate_length, self.accumulate_length))

        self.accumulated_steppable_image[150:350, 50:250] = np.ones((200, 200)) * 255

        tmp_model_steppable = cnn_models.cnn_steppable((500, 300, 1), self.checkpoint_path)
        self.model_steppable = K.function([tmp_model_steppable.input], [tmp_model_steppable.output])
        tmp_model_height = cnn_models.cnn_height((500, 300, 1), self.checkpoint_path)
        self.model_height = K.function([tmp_model_height.input], [tmp_model_height.output])
        tmp_model_pose = cnn_models.cnn_pose((500, 300, 1), self.checkpoint_path)
        self.model_pose = K.function([tmp_model_pose.input], [tmp_model_pose.output])

        self.lock = threading.Lock()

        self.height_publisher = rospy.Publisher('AutoStabilizerROSBridge/landing_height', OnlineFootStep, queue_size=1)
        self.landing_pose_publisher = rospy.Publisher('landing_pose_marker', Marker, queue_size = 1)
        self.polygon_publisher = rospy.Publisher('output_polygon', PolygonArray, queue_size=1)
        self.convex_publisher = rospy.Publisher('combined_meshed_polygons', PolygonArray, queue_size=1)
        self.region_publisher = rospy.Publisher('AutoStabilizerROSBridge/steppable_region', SteppableRegion, queue_size=1)
        self.visualized_image_publisher = rospy.Publisher('steppable_image_output', Image, queue_size=1)
        self.visualized_trimmed_image_publisher = rospy.Publisher('trimmed_image_output', Image, queue_size=1)
        self.step_heightmap_publisher = rospy.Publisher('step_heightmap/output', Image, queue_size=1)
        self.step_heightmap_config_publisher = rospy.Publisher('step_heightmap/output/config', HeightmapConfig, queue_size=1)

        rospy.Subscriber('rt_filtered_current_heightmap/output/config', HeightmapConfig, self.heightmapConfigCallback, queue_size=1)
        rospy.Subscriber('rt_filtered_current_heightmap/output', Image, self.heightmapCallback, queue_size=1, buff_size=2**24)
        rospy.Subscriber('AutoStabilizerROSBridge/landing_target', OnlineFootStep, self.targetCallback, queue_size=1)

    def heightmapConfigCallback(self, msg):
        self.heightmap_config_flag = True
        self.heightmap_minx = math.floor(msg.min_x * 100)
        self.heightmap_miny = math.floor(msg.min_y * 100)
        self.heightmap_maxx = math.floor(msg.max_x * 100)
        self.heightmap_maxy = math.floor(msg.max_y * 100)

    def heightmapCallback(self, msg):
        begin_time = rospy.Time.now()
        if self.debug_output:
            print("stamp_begin", (begin_time - msg.header.stamp).secs, "s", (int)((begin_time - msg.header.stamp).nsecs / 1000000), "ms")

        if not self.heightmap_config_flag:
            return

        #input_image = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width, -1)
        input_image = ros_numpy.numpify(msg)
        median_image = cv2.medianBlur(input_image[:, :, 0], 5)
        median_image = median_image - (median_image<-1e5)*median_image + (median_image<-1e5) * cv2.blur(median_image - (median_image<-1e5)*median_image, (9, 9)) / cv2.blur((median_image>-1e5).astype(float), (9, 9))
        np.nan_to_num(median_image, copy=False, nan=-1e10)

        tmp, update_pixel = cv2.threshold(median_image, -1e5, 1, cv2.THRESH_BINARY)
        update_pixel = np.uint8(update_pixel)
        cv2.rectangle(update_pixel, (0, 0), (update_pixel.shape[1]-1, update_pixel.shape[0]-1), 0, thickness=1)
        update_pixel = cv2.erode(update_pixel, np.ones((3, 3), np.uint8), iterations = (int)(self.update_pixel_erode))

        a_time = rospy.Time.now()
        if self.debug_output:
            print("begin_a", (a_time - begin_time).secs, "s", (int)((a_time - begin_time).nsecs / 1000000), "ms")
        #steppable_regionの処理

        cnn_steppable_img = median_image.reshape((1, msg.height, msg.width, 1))
        cnn_steppable_img = np.array(self.model_steppable([cnn_steppable_img])[0])
        np.nan_to_num(cnn_steppable_img, copy=False)
        cnn_steppable_img = np.argmax(cnn_steppable_img, axis=3)
        cnn_steppable_img = cnn_steppable_img.reshape((cnn_steppable_img.shape[1], cnn_steppable_img.shape[2], 1))
        cnn_steppable_img = np.uint8(cnn_steppable_img*255)
        #strideがある場合resize, resizeの有無でshapeが変わるので注意
        cnn_steppable_img = cv2.resize(cnn_steppable_img, (cnn_steppable_img.shape[1]*2, cnn_steppable_img.shape[0]*2))
        tmp_img = np.zeros((msg.height, msg.width))
        tmpy = math.floor((msg.height - cnn_steppable_img.shape[0])/2)
        tmpx = math.floor((msg.width - cnn_steppable_img.shape[1])/2)
        tmp_img[tmpy : tmpy+cnn_steppable_img.shape[0], tmpx : tmpx+cnn_steppable_img.shape[1]] = cnn_steppable_img.copy()
        cnn_steppable_img = tmp_img.copy()


        b_time = rospy.Time.now()
        if self.debug_output:
            print("a_b", (b_time - a_time).secs, "s", (int)((b_time - a_time).nsecs / 1000000), "ms")
        #着地位置姿勢の処理
        
        cnn_height_img = median_image.reshape((1, msg.height, msg.width, 1))
        cnn_height_img = np.array(self.model_height([cnn_height_img])[0])
        np.nan_to_num(cnn_height_img, copy=False)
        cnn_height_img = cnn_height_img.reshape((cnn_height_img.shape[1], cnn_height_img.shape[2], 1))
        #strideがある場合resize
        cnn_height_img = cv2.resize(cnn_height_img, (cnn_height_img.shape[1]*2, cnn_height_img.shape[0]*2))
        tmp_img = np.zeros((msg.height, msg.width))
        tmpy = math.floor((msg.height - cnn_height_img.shape[0])/2)
        tmpx = math.floor((msg.width - cnn_height_img.shape[1])/2)
        tmp_img[tmpy : tmpy+cnn_height_img.shape[0], tmpx : tmpx+cnn_height_img.shape[1]] = cnn_height_img.copy()
        cnn_height_img = tmp_img.copy()


        cnn_pose_img = median_image.reshape((1, msg.height, msg.width, 1))
        cnn_pose_img = np.array(self.model_pose([cnn_pose_img])[0])
        np.nan_to_num(cnn_pose_img, copy=False)
        cnn_pose_img = cnn_pose_img.reshape((cnn_pose_img.shape[1], cnn_pose_img.shape[2], 2))
        #strideがある場合resize
        cnn_pose_img = cv2.resize(cnn_pose_img, (cnn_pose_img.shape[1]*2, cnn_pose_img.shape[0]*2))
        tmp_img = np.zeros((msg.height, msg.width, 2))
        tmpy = math.floor((msg.height - cnn_pose_img.shape[0])/2)
        tmpx = math.floor((msg.width - cnn_pose_img.shape[1])/2)
        tmp_img[tmpy : tmpy+cnn_pose_img.shape[0], tmpx : tmpx+cnn_pose_img.shape[1]] = cnn_pose_img.copy()
        cnn_pose_img = tmp_img.copy()

        c_time = rospy.Time.now()
        if self.debug_output:
            print("b_c", (c_time - b_time).secs, "s", (int)((c_time - b_time).nsecs / 1000000), "ms")

        #蓄積
        with self.lock:
            try:
                self.listener.waitForTransform(self.fixed_frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(1.0))
                p, q = self.listener.lookupTransform(self.fixed_frame, msg.header.frame_id, msg.header.stamp)
                R = quaternion.as_rotation_matrix(np.quaternion(q[3], q[0], q[1], q[2]))
                self.center_H[:3, 3] = np.array(p)*100
                self.center_H[:3, :3] = R
                if np.linalg.norm(self.prev_H[:3, 3]) <= 1e-10:
                    self.prev_H = self.center_H.copy()
                    return
            except:
                print("tf error")
                return

            img_H = np.identity(4)
            img_H[:3, 3] = np.array([-self.accumulate_center_x, -self.accumulate_center_y, 0])
            tmp_H = np.linalg.inv(self.center_H @ img_H) @ (self.prev_H @ img_H)
            trans_cv = np.delete(tmp_H[:2, :], 2, 1)
            self.prev_H = self.center_H.copy()

            self.accumulated_steppable_image = cv2.warpAffine(self.accumulated_steppable_image, trans_cv, (self.accumulated_steppable_image.shape[1], self.accumulated_steppable_image.shape[0]), flags=cv2.INTER_NEAREST)
            self.accumulated_height_image = cv2.warpAffine(self.accumulated_height_image, trans_cv, (self.accumulated_height_image.shape[1], self.accumulated_height_image.shape[0]))
            self.accumulated_pose_image = cv2.warpAffine(self.accumulated_pose_image, trans_cv, (self.accumulated_pose_image.shape[1], self.accumulated_pose_image.shape[0]))
            self.accumulated_yaw_image = cv2.warpAffine(self.accumulated_yaw_image, trans_cv, (self.accumulated_yaw_image.shape[1], self.accumulated_yaw_image.shape[0]))

            tmp_x = self.accumulate_center_x + self.heightmap_minx
            tmp_y = self.accumulate_center_y + self.heightmap_miny
            self.accumulated_steppable_image[tmp_y : tmp_y + msg.height, tmp_x : tmp_x + msg.width] = cnn_steppable_img * update_pixel + self.accumulated_steppable_image[tmp_y : tmp_y + msg.height, tmp_x : tmp_x + msg.width] * (1 - update_pixel)
            #高さだけ平行移動対応する
            diff_img = cnn_height_img * update_pixel - self.accumulated_height_image[tmp_y : tmp_y + msg.height, tmp_x : tmp_x + msg.width] * update_pixel
            diff_value = np.median(diff_img[(update_pixel*(np.abs(diff_img)<0.1))>0.5]) #update_pixelかつ誤差が0.1m以下のものの中央値
            self.accumulated_height_image += diff_value
            self.accumulated_height_image[tmp_y : tmp_y + msg.height, tmp_x : tmp_x + msg.width] = cnn_height_img * update_pixel + self.accumulated_height_image[tmp_y : tmp_y + msg.height, tmp_x : tmp_x + msg.width] * (1 - update_pixel)
            self.accumulated_pose_image[tmp_y : tmp_y + msg.height, tmp_x : tmp_x + msg.width] = cnn_pose_img * np.dstack((update_pixel, update_pixel)) + self.accumulated_pose_image[tmp_y : tmp_y + msg.height, tmp_x : tmp_x + msg.width] * (1 - np.dstack((update_pixel, update_pixel)))
            current_yaw_img = np.ones((msg.height, msg.width)) * np.arctan2(self.center_H[1, 0], self.center_H[0, 0])
            self.accumulated_yaw_image[tmp_y : tmp_y + msg.height, tmp_x : tmp_x + msg.width] = current_yaw_img * update_pixel + self.accumulated_yaw_image[tmp_y : tmp_y + msg.height, tmp_x : tmp_x + msg.width] * (1 - update_pixel)

            d_time = rospy.Time.now()
            if self.debug_output:
                print("c_d", (d_time - c_time).secs, "s", (int)((d_time - c_time).nsecs / 1000000), "ms")

            #ここで拡大縮小、輪郭抽出等
            trimmed_image = np.uint8(self.accumulated_steppable_image[self.accumulate_center_y - self.trim_center_y : self.accumulate_center_y - self.trim_center_y + self.trim_length, self.accumulate_center_x - self.trim_center_x : self.accumulate_center_x - self.trim_center_x + self.trim_length].copy())
            trimmed_image = cv2.morphologyEx(trimmed_image, cv2.MORPH_OPEN, np.ones((3, 3)))
            trimmed_image = cv2.morphologyEx(trimmed_image, cv2.MORPH_CLOSE, np.ones((5, 5)))
            trimmed_image = cv2.erode(trimmed_image, np.ones((5, 5), np.uint8))
            trimmed_image = cv2.morphologyEx(trimmed_image, cv2.MORPH_OPEN, np.ones((5, 5)))
            trimmed_image = cv2.dilate(trimmed_image, np.ones((3, 3), np.uint8))

            visualized_trimmed_image = np.zeros((trimmed_image.shape[0], trimmed_image.shape[1], 3), dtype=np.uint8)
            visualized_trimmed_image[:, :, 0] = trimmed_image.copy()

            contours, hierarchy = cv2.findContours(trimmed_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


            #size_threshold = 5
            #approx_vector = []
            #for i in range(len(contours)):
            #    if cv2.contourArea(contours[i]) > size_threshold:
            #        approx = cv2.approxPolyDP(contours[i], 1.5, True)
            #        if len(approx) >= 3:
            #            tmp = []
            #            for p in approx:
            #                tmp.append(p2t.Point(p[0, 0], p[0, 1]))
            #            approx_vector.append(tmp)

            #tri_list = []
            #for i in range(len(approx_vector)):
            #    if hierarchy[0, i, 3] != -1:
            #        continue
            #    cdt = p2t.CDT(approx_vector[i])
            #    j = hierarchy[0, i, 2] #first hole
            #    while j != -1:
            #        cdt.add_hole(approx_vector[j])
            #        j = hierarchy[0, j, 0] #next hole
            #    tri_list.extend(cdt.triangulate())

            size_threshold = 5
            tri_list = []
            for i in range(len(contours)):
                if hierarchy[0, i, 3] != -1: #穴は後で
                    continue
                if cv2.contourArea(contours[i]) > size_threshold:
                    approx = cv2.approxPolyDP(contours[i], 1.5, True)
                    if len(approx) >= 3:
                        tmp = []
                        #print("shape")
                        for p in approx:
                            tmp.append(p2t.Point(p[0, 0], p[0, 1]))
                            #print (p[0, 0], p[0, 1])
                            cv2.circle(visualized_trimmed_image, (p[0, 0], p[0, 1]), 2, (0, 255, 0), -1)
                        cdt = p2t.CDT(tmp)
                        j = hierarchy[0, i, 2] #first hole
                        while j != -1:
                            if cv2.contourArea(contours[j]) > size_threshold:
                                approx_hole = cv2.approxPolyDP(contours[j], 1.5, True)
                                if len(approx_hole) >= 3:
                                    tmp_hole = []
                                    #print("hole")
                                    for ph in approx_hole:
                                        tmp_hole.append(p2t.Point(ph[0, 0], ph[0, 1]))
                                        #print (ph[0, 0], ph[0, 1])
                                        cv2.circle(visualized_trimmed_image, (ph[0, 0], ph[0, 1]), 2, (0, 0, 255), -1)
                                    cdt.add_hole(tmp_hole)
                            j = hierarchy[0, j, 0] #next hole
                        tri_list.extend(cdt.triangulate())

            #tri_list = []
            #for i in range(len(approx_vector)):
            #    if approx_hierarchy[i][3] != -1:
            #        continue
            #    cdt = p2t.CDT(approx_vector[i])
            #    j = approx_hierarchy[i][2] #first hole
            #    while j != -1:
            #        cdt.add_hole(approx_vector[j])
            #        j = approx_hierarchy[j][0] #next hole
            #    tri_list.extend(cdt.triangulate())

            if len(tri_list) == 0:
                print("error")
                return

            self.convex_list = []
            now_convex = []
#i=0
            if self.t2pPointEqual(tri_list[0].b, tri_list[1].a) or self.t2pPointEqual(tri_list[0].b, tri_list[1].b) or self.t2pPointEqual(tri_list[0].b, tri_list[1].c):
                now_convex = [tri_list[0].a, tri_list[0].b, tri_list[0].c]
            else:
                now_convex = [tri_list[0].b, tri_list[0].c, tri_list[0].a]
            index = 1


            for i in range(1, len(tri_list)):
                #for i in range(len(now_convex)):
                #    cv2.line(img, (int(now_convex[i-1].x), int(now_convex[i-1].y)), (int(now_convex[i].x), int(now_convex[i].y)), color=(0, 200, 100), thickness=3)
                #cv2.imshow("test", img)
                #cv2.waitKey(1000)

                tmp_tri = []
                if self.t2pPointEqual(now_convex[index], tri_list[i].a):
                    tmp_tri = [tri_list[i].a, tri_list[i].b, tri_list[i].c]
                elif self.t2pPointEqual(now_convex[index], tri_list[i].b):
                    tmp_tri = [tri_list[i].b, tri_list[i].c, tri_list[i].a]
                elif self.t2pPointEqual(now_convex[index], tri_list[i].c):
                    tmp_tri = [tri_list[i].c, tri_list[i].a, tri_list[i].b]
                else:
                    self.convex_list.append(copy.deepcopy(now_convex))
                    if i == len(tri_list)-1:
                        self.convex_list.append([tri_list[i].a, tri_list[i].b, tri_list[i].c])
                    elif self.t2pPointEqual(tri_list[i].b, tri_list[i+1].a) or self.t2pPointEqual(tri_list[i].b, tri_list[i+1].b) or self.t2pPointEqual(tri_list[i].b, tri_list[i+1].c):
                        now_convex = [tri_list[i].a, tri_list[i].b, tri_list[i].c]
                    else:
                        now_convex = [tri_list[i].b, tri_list[i].c, tri_list[i].a]
                    index = 1
                    continue

                tmp_index = index+2 if index+2 != len(now_convex) else 0
                if self.t2pPointEqual(now_convex[index-1], tmp_tri[1]) and np.cross([now_convex[index-1].x - now_convex[index-2].x, now_convex[index-1].y - now_convex[index-2].y], [tmp_tri[2].x - now_convex[index-2].x, tmp_tri[2].y - now_convex[index-2].y]) > 0 and np.cross([now_convex[index].x - now_convex[index+1].x, now_convex[index].y - now_convex[index+1].y], [tmp_tri[2].x - now_convex[index+1].x, tmp_tri[2].y - now_convex[index+1].y]) < 0:
                    now_convex.insert(index, tmp_tri[2])
                elif self.t2pPointEqual(now_convex[index+1], tmp_tri[2]) and np.cross([now_convex[index].x - now_convex[index-1].x, now_convex[index].y - now_convex[index-1].y], [tmp_tri[1].x - now_convex[index-1].x, tmp_tri[1].y - now_convex[index-1].y]) > 0 and np.cross([now_convex[index+1].x - now_convex[tmp_index].x, now_convex[index+1].y - now_convex[tmp_index].y], [tmp_tri[1].x - now_convex[tmp_index].x, tmp_tri[1].y - now_convex[tmp_index].y]) < 0:
                    now_convex.insert(index+1, tmp_tri[1])
                    index += 1
                else:
                    self.convex_list.append(copy.deepcopy(now_convex))
                    if i == len(tri_list)-1:
                        self.convex_list.append([tri_list[i].a, tri_list[i].b, tri_list[i].c])
                    elif self.t2pPointEqual(tri_list[i].b, tri_list[i+1].a) or self.t2pPointEqual(tri_list[i].b, tri_list[i+1].b) or self.t2pPointEqual(tri_list[i].b, tri_list[i+1].c):
                        now_convex = [tri_list[i].a, tri_list[i].b, tri_list[i].c]
                    else:
                        now_convex = [tri_list[i].b, tri_list[i].c, tri_list[i].a]
                    index = 1
                    continue
                if i == len(tri_list)-1:
                    self.convex_list.append(copy.deepcopy(now_convex))

        e_time = rospy.Time.now()
        if self.debug_output:
            print("d_e", (e_time - d_time).secs, "s", (int)((e_time - d_time).nsecs / 1000000), "ms")

        polygon_msg = PolygonArray()
        polygon_msg.header = copy.deepcopy(msg.header)
        polygon_msg.header.frame_id = self.fixed_frame
        for p in tri_list:
            ps = PolygonStamped()
            ps.header = polygon_msg.header
            p32 = Point32()
            tmpa = np.array([p.a.x - self.trim_center_x, p.a.y - self.trim_center_y, self.accumulated_height_image[math.floor(p.a.y - self.trim_center_y + self.accumulate_center_y), math.floor(p.a.x - self.trim_center_x + self.accumulate_center_x)] * 100.0, 1])
            if tmpa[2] < -1e+10:
                tmpa[2] = 0
            tmpa = self.center_H @ tmpa
            tmpa = tmpa * 0.01
            p32.x = tmpa[0]
            p32.y = tmpa[1]
            p32.z = tmpa[2]
            ps.polygon.points.append(copy.deepcopy(p32))

            tmpb = np.array([p.b.x - self.trim_center_x, p.b.y - self.trim_center_y, self.accumulated_height_image[math.floor(p.b.y - self.trim_center_y + self.accumulate_center_y), math.floor(p.b.x - self.trim_center_x + self.accumulate_center_x)] * 100.0, 1])
            if tmpb[2] < -1e+10:
                tmpb[2] = 0
            tmpb = self.center_H @ tmpb
            tmpb = tmpb * 0.01
            p32.x = tmpb[0]
            p32.y = tmpb[1]
            p32.z = tmpb[2]
            ps.polygon.points.append(copy.deepcopy(p32))

            tmpc = np.array([p.c.x - self.trim_center_x, p.c.y - self.trim_center_y, self.accumulated_height_image[math.floor(p.c.y - self.trim_center_y + self.accumulate_center_y), math.floor(p.c.x - self.trim_center_x + self.accumulate_center_x)] * 100.0, 1])
            if tmpc[2] < -1e+10:
                tmpc[2] = 0
            tmpc = self.center_H @ tmpc
            tmpc = tmpc * 0.01
            p32.x = tmpc[0]
            p32.y = tmpc[1]
            p32.z = tmpc[2]
            ps.polygon.points.append(copy.deepcopy(p32))
            polygon_msg.polygons.append(copy.deepcopy(ps))

        self.polygon_publisher.publish(polygon_msg)
        
        convex_msg = PolygonArray()
        convex_msg.header = copy.deepcopy(msg.header)
        convex_msg.header.frame_id = self.fixed_frame
        for v in self.convex_list:
            ps = PolygonStamped()
            ps.header = convex_msg.header
            p32 = Point32()
            for p in v:
                tmp = np.array([p.x - self.trim_center_x, p.y - self.trim_center_y, self.accumulated_height_image[math.floor(p.y - self.trim_center_y + self.accumulate_center_y), math.floor(p.x - self.trim_center_x + self.accumulate_center_x)] * 100.0, 1])
                if tmp[2] < -1e+10:
                    tmp[2] = 0
                tmp = self.center_H @ tmp
                tmp = tmp * 0.01
                p32.x = tmp[0]
                p32.y = tmp[1]
                p32.z = tmp[2]
                ps.polygon.points.append(copy.deepcopy(p32))
            convex_msg.polygons.append(copy.deepcopy(ps))
        self.convex_publisher.publish(convex_msg)

        visualized_image = np.zeros((self.accumulated_steppable_image.shape[0], self.accumulated_steppable_image.shape[1], 3), dtype=np.uint8)
        visualized_image[:, :, 0] = self.accumulated_steppable_image.copy()
        visualized_image[self.accumulate_center_y + self.heightmap_miny : self.accumulate_center_y + self.heightmap_miny + msg.height, self.accumulate_center_x + self.heightmap_minx : self.accumulate_center_x + self.heightmap_minx + msg.width, 1] = update_pixel.copy() * 100
        visualized_image[self.accumulate_center_y + self.heightmap_miny : self.accumulate_center_y + self.heightmap_miny + msg.height, self.accumulate_center_x + self.heightmap_minx : self.accumulate_center_x + self.heightmap_minx + msg.width, 2] = (median_image.copy() * 255 * 1.0 + 80).astype(np.uint8)
        visualized_image = cv2.flip(visualized_image, 0)
        self.visualized_image_publisher.publish(ros_numpy.msgify(Image, visualized_image, encoding='bgr8'))
        visualized_trimmed_image = cv2.flip(visualized_trimmed_image, 0)

        trimmed_msg = ros_numpy.msgify(Image, visualized_trimmed_image, encoding='bgr8')
        trimmed_msg.header = copy.deepcopy(msg.header)
        self.visualized_trimmed_image_publisher.publish(trimmed_msg)

        step_heightmap_image = np.zeros(self.accumulated_height_image.shape)
        step_heightmap_image = np.dstack((self.accumulated_height_image, step_heightmap_image))
        step_heightmap_msg = ros_numpy.msgify(Image, step_heightmap_image.astype(np.float32), encoding='32FC2')
        step_heightmap_msg.header = copy.deepcopy(msg.header)
        self.step_heightmap_publisher.publish(step_heightmap_msg)

        step_heightmap_config_msg = HeightmapConfig()
        step_heightmap_config_msg.max_x = 0.01 * (self.accumulate_length - self.accumulate_center_x)
        step_heightmap_config_msg.min_x = 0.01 * (-self.accumulate_center_x)
        step_heightmap_config_msg.max_y = 0.01 * (self.accumulate_length - self.accumulate_center_y)
        step_heightmap_config_msg.min_y = 0.01 * (-self.accumulate_center_y)
        #step_heightmap_config_msg.max_x = 1.0
        #step_heightmap_config_msg.min_x = 0.0
        #step_heightmap_config_msg.max_y = 1.0
        #step_heightmap_config_msg.min_y = 0.0
        self.step_heightmap_config_publisher.publish(step_heightmap_config_msg)

        end_time = rospy.Time.now()


        if self.debug_output:
            print("e_end", (end_time - e_time).secs, "s", (int)((end_time - e_time).nsecs / 1000000), "ms")
            print("                      all_time", (end_time - begin_time).secs, "s", (int)((end_time - begin_time).nsecs / 1000000), "ms")
            print("                all_stamp_time", (end_time - msg.header.stamp).secs, "s", (int)((end_time - msg.header.stamp).nsecs / 1000000), "ms")

        #cv2.imshow('test1',self.accumulated_height_image[:, :]*3.0 + 0.2)
        #cv2.imshow('test2',self.accumulated_steppable_image)
        #cv2.imshow('test2',visualized_image)
        #cv2.imshow('test1',trimmed_image)
        #cv2.imshow('test2',(1 - update_pixel) * 255)
        #cv2.imshow('test3',cnn_height_img*3.0 + 0.2)
        #cv2.waitKey(1)

    def targetCallback(self, msg):
        if msg.l_r >= 2:
            return
        target_frame = "lleg_end_coords" if msg.l_r%2 == 1 else "rleg_end_coords"
        try:
            self.listener.waitForTransform(self.fixed_frame, target_frame, msg.header.stamp, rospy.Duration(1.0))
            p, q = self.listener.lookupTransform(self.fixed_frame, target_frame, msg.header.stamp)
            R = quaternion.as_rotation_matrix(np.quaternion(q[3], q[0], q[1], q[2]))
            self.cur_foot_pos = np.array(p) * 100
            self.cur_foot_rot = R
            self.cur_foot_rot_ground = self.calcFootRotFromNormal(R, np.array([0, 0, 1]))
            self.next_foot_pos = self.cur_foot_pos + self.cur_foot_rot_ground @ (np.array([msg.x, msg.y, msg.z]) * 100)
        except:
            print("tf error")
            return

        with self.lock:
            img_H = np.identity(4)
            img_H[:3, 3] = np.array([-self.accumulate_center_x, -self.accumulate_center_y, 0])
            cur_tmp = np.linalg.inv(self.center_H @ img_H)[:3, :] @ np.append(self.cur_foot_pos, 1)
            next_tmp = np.linalg.inv(self.center_H @ img_H)[:3, :] @ np.append(self.next_foot_pos, 1)
            self.cur_foot_pos[2] = self.accumulated_height_image[math.floor(cur_tmp[1]), math.floor(cur_tmp[0])] * 100
            self.next_foot_pos[2] = self.accumulated_height_image[math.floor(next_tmp[1]), math.floor(next_tmp[0])] * 100

            cur_foot_ground_H = np.identity(4)
            cur_foot_ground_H[:3, :3] = self.cur_foot_rot_ground
            cur_foot_ground_H[:3, 3] = self.cur_foot_pos
            sr = SteppableRegion()
            sr.header = copy.deepcopy(msg.header)
            sr.header.frame_id = target_frame
            sr.l_r = msg.l_r

            for v in self.convex_list:
                ps = PolygonStamped()
                ps.header = sr.header
                p32 = Point32()
                for p in v:
                    tmp = np.array([p.x - self.trim_center_x, p.y - self.trim_center_y, self.accumulated_height_image[math.floor(p.y - self.trim_center_y + self.accumulate_center_y), math.floor(p.x - self.trim_center_x + self.accumulate_center_x)] * 100.0, 1])
                    if tmp[2] < -1e+10:
                        tmp[2] = 0
                    tmp = np.linalg.inv(cur_foot_ground_H) @ self.center_H @ tmp
                    tmp = tmp * 0.01
                    p32.x = tmp[0]
                    p32.y = tmp[1]
                    p32.z = tmp[2]
                    ps.polygon.points.append(copy.deepcopy(p32))
                sr.polygons.append(copy.deepcopy(ps))

            ps = OnlineFootStep()
            ps.header = copy.deepcopy(msg.header)
            ps.header.frame_id = target_frame
            ps.l_r = msg.l_r

            tmp_vecx = [1.0, 0.0, self.accumulated_pose_image[math.floor(next_tmp[1]), math.floor(next_tmp[0]), 0]]
            tmp_vecy = [0.0, 1.0, self.accumulated_pose_image[math.floor(next_tmp[1]), math.floor(next_tmp[0]), 1]]
            tmp_vecz = np.cross(tmp_vecx, tmp_vecy)
            tmp_vecz = tmp_vecz / np.linalg.norm(tmp_vecz)
            if tmp_vecz[2] < 0.8:
                print("pose error")
                tmp_vecz[0] = 0.
                tmp_vecz[1] = 0.
                tmp_vecz[2] = 1.
            tmp_vecz = self.cur_foot_rot_ground.transpose() @ quaternion.as_rotation_matrix(quaternion.from_rotation_vector(np.array([0, 0, self.accumulated_yaw_image[math.floor(next_tmp[1]), math.floor(next_tmp[0])]]))) @ tmp_vecz

            ps.nx = tmp_vecz[0]
            ps.ny = tmp_vecz[1]
            ps.nz = tmp_vecz[2]

            tmp_pos = (self.cur_foot_rot_ground.transpose() @ (self.next_foot_pos - self.cur_foot_pos)) * 0.01
            tmp_pos[2] = tmp_pos[2] if math.fabs(tmp_pos[2]) < 0.4 else 0
            ps.x = tmp_pos[0]
            ps.y = tmp_pos[1]
            ps.z = tmp_pos[2]


            # publish pose msg for visualize
            start_pos = self.cur_foot_rot.transpose() @ self.cur_foot_rot_ground @ tmp_pos
            end_pos = self.cur_foot_rot.transpose() @ self.cur_foot_rot_ground @ (tmp_pos + 0.3 * tmp_vecz)
            pose_msg = Marker()
            pose_msg.header = copy.deepcopy(ps.header)
            pose_msg.ns = "landing_pose"
            pose_msg.id = 0
            pose_msg.type = Marker.ARROW
            pose_msg.action = Marker.ADD
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
            pose_msg.color.r = 0.8
            pose_msg.color.g = 0.0
            pose_msg.color.b = 1.0
            pose_msg.color.a = 1.0
            pose_msg.scale.x = 0.03
            pose_msg.scale.y = 0.05
            pose_msg.scale.z = 0.07
            pose_msg.pose.orientation.w = 1.0
        self.region_publisher.publish(sr)
        self.height_publisher.publish(ps)
        self.landing_pose_publisher.publish(pose_msg)

    def calcFootRotFromNormal(self, orig_R, n):
        en = n / np.linalg.norm(n)
        ex = np.array([1, 0, 0])
        xv1 = orig_R @ ex
        xv1 = xv1 - xv1 @ en * en
        xv1 = xv1 / np.linalg.norm(xv1)
        yv1 = np.cross(en, xv1)
        ret = np.zeros((3, 3))
        ret[:, 0] = xv1
        ret[:, 1] = yv1
        ret[:, 2] = en
        return ret

    def t2pPointEqual(self, p, q):
        return p.x == q.x and p.y == q.y

if __name__=='__main__':
    rospy.init_node('legpose_pub')
    steppable_region_publisher = SteppableRegionPublisher()
    rospy.spin()
