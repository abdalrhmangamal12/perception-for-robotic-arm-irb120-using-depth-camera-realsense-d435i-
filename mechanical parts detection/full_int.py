#!/usr/bin/env python
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import pyrealsense2 as rs
import numpy as np
import rospy
from python_test.srv import *
from geometry_msgs.msg import PointStamped
from detection_helpers_number import number_Detector
from detection_helpers_parts import parts_Detector

from std_msgs.msg import String
from python_test.srv import RealSense, RealSenseRequest, RealSenseResponse
c=0 

# Configure depth and color streams

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile=pipeline.start(config)

sensor = profile.get_device().query_sensors()[0]
sensor.set_option(rs.option.enable_auto_exposure, True)
#Increase gain for high light conditions
sensor.set_option(rs.option.gain, -3)


align = rs.align(rs.stream.color)

frames = pipeline.wait_for_frames()
frames = pipeline.wait_for_frames()
frames = pipeline.wait_for_frames()
depth_frame =0
parts_informdict =0
number_informdict =0


def number_callback(mm):
    global number_informdict 
    global c
    response=RealSenseResponse()


    response.res= str(number_informdict)
    print(response)
    c =1
    return response


def parts_callback(data):
    global parts_informdict
    response=RealSenseResponse()
    response.res= str(parts_informdict)
    print(response)
    return response


if __name__== '__main__':



   rospy.init_node("traker")
   rospy.Service('parts_service',RealSense,parts_callback)
   rospy.Service('number_service',RealSense,number_callback)
   
   number_detect=number_Detector()
   parts_detect = parts_Detector()
   while True:
         # Read a frame from the video stream
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        profile = aligned_frames.get_profile()


        color_frame = aligned_frames.get_color_frame()
       # global depth_frame
        depth_frame = aligned_frames.get_depth_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()
     #   print(depth_intrinsics)
        spatial = rs.spatial_filter()
        #spatial.set_option(rs.option.filter_magnitude, 5)
        #spatial.set_option(rs.option.filter_smooth_alpha, 1)
        #spatial.set_option(rs.option.filter_smooth_delta, 50)
        
        filtered_depth = spatial.process(depth_frame)

        parts_img = np.asanyarray(color_frame.get_data())
        number_img = parts_img.copy()
        depth_image = np.asanyarray(filtered_depth.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


        
       # parts_informdict=parts_detect.detect(img,depth_intrin,depth_frame)
        parts_informdict=parts_detect.detect(parts_img,depth_intrin,depth_frame)

      #  if c ==0:
         #   number_informdict=number_detect.detect(number_img,depth_intrin,depth_frame)
       # print(number_informdict ,'efgfhghjh')
