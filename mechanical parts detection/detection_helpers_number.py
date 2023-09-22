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
import rospy
from sensor_msgs.msg import Image
from ros_numpy import msgify

#from python_test.srv import RealSense, RealSenseRequest, RealSenseResponse
   

gears_informdict={} 




class number_Detector:
    def __init__(self):

        self.weights = '/home/abdo_gamal/gp_ws/src/python_test/scripts/yolov7-new/number detection/yolov7number/runs/train/exp/weights/best.pt'
        self.source = 'inference/images'
        self.img_size = 640
        self.conf_thres = 0.73
        self.iou_thres = 0.6
        self.device = select_device("0" if torch.cuda.is_available() else 'cpu')
        self.view_img = False
        self.save_txt =False
        self.save_conf = False
        self.nosave =  False
        self.classes = None
        self.agnostic_nms =  False
        self.augment =  False
        self.update = False
        self.project = 'runs/detect'
        self.names = 'exp'
        self.exist_ok = False
        self.no_trace = False
        self.model =0
        self.half=0
        self.imgsz=640
        self.trace =True
        self.vertics_rec =[]
        self.labels = []
    # Directories
        save_dir = Path(increment_path(Path(self.project) / self.names, exist_ok=self.exist_ok))  # increment run
        (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
        set_logging()

        half = self.device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.img_size, s=stride)  # check img_size

        if self.trace:
            self.model = TracedModel(self.model, self.device, self.imgsz)

        if self.half:
           self.model.half()  # to FP16

    # Second-stage classifier
        classify = False
        if classify:
           modelc = load_classifier(name='resnet101', n=2)  # initialize
           modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

    # Set Dataloader
        vid_path, vid_writer = None, None
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

        if webcam:
           view_img = check_imshow()
           cudnn.benchmark = True  # set True to speed up constant image size inference
           dataset = LoadStreams(self.source, img_size=self.img_size, stride=stride)
        else:
           dataset = LoadImages(self.source, img_size=self.img_size, stride=stride)

    # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    # Run inference
        if self.device.type != 'cpu':
           self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w  = self.imgsz
        self.old_img_h =self.old_img_w
        self.old_img_b = 1

    @torch.no_grad()
    def pointconvert(self,depth_intrin,numb_dictionary,depth_frame):
    
        #Use pixel value of  depth-aligned color image to get 3D axes
        gears_with_depth_point=[]
        for i  in numb_dictionary.values():
            color_point = [i[0],i[1]]
            
        
            if color_point[1]==480:
                color_point[1]=479
            if color_point[0]==640:
                color_point[0]=639
  

            depth = depth_frame.get_distance(round(color_point[0]) , round(color_point[1]))
            dx,dy, dz = rs.rs2_deproject_pixel_to_point(depth_intrin, color_point , depth) 
         
            print("depth from point to camera ", depth)
            print(f'x is {dx} and y is {dy} and z is {dz}')
            gears_with_depth_point.append([dx,dy,dz])
        numb_inform=dict(zip(numb_dictionary.keys(),gears_with_depth_point))
        return numb_inform
    


    def detect(self,image,depth_intrin,depth_frame):
   
        # Letterbox
        self.vertics_rec.clear() 
        self.labels.clear() 
    
        im0 = image
        
        img = im0[np.newaxis, :, :, :]        
      
        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)


        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
#        if classify:
#            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image

#            p = Path(p)  # to Path
#            save_path = str(save_dir / p.name)  # img.jpg
#            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                 #   s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string






                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{self.names[c]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)
                    #plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)
                    c=torch.tensor(xyxy)
                  
                    self.labels.append(str(label)[0:-5])
                    self.vertics_rec.append(c.numpy())
            vertics_rec_array=np.array(self.vertics_rec)
          #  print(vertics_rec_array)
           # print(np.shape(vertics_rec_array))
            centre_circle=[]

            m=vertics_rec_array.shape
            for i in range(0,m[0]):
                 x_circle=(((vertics_rec_array[i][0])+(vertics_rec_array[i][2]))/2)
                 y_circle=(((vertics_rec_array[i][1])+(vertics_rec_array[i][3]))/2)
                 centre_circle.append((round(x_circle),round(y_circle)))
                 cv2.circle(im0,(round(x_circle),round(y_circle)),3,(0,255,0),2)

            gears_inform=dict(zip(self.labels,centre_circle))
               


                        
           # gears_informarray=np.array(list(gears_inform.items()))
        #     # convert 2d point to 3d 3d from convert file using  function pointconvert
            global gears_informdict
            gears_informdict= self.pointconvert(depth_intrin , gears_inform,depth_frame)
           # print(gears_informdict)
       #sssss     image_msg = msgify(Image, im0, encoding='rgb8')
       #     rate = rospy.Rate(1)
       #     pub = rospy.Publisher('number_topic', Image, queue_size=10)
          #  pub.publish(image_msg)
        #    rate.sleep()  
            cv2.imshow("number_detection", im0)
           # cv2.imshow(" result", depth_frame)
    
            # cv2.imshow("Recognition result depth",depth_colormap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            return gears_informdict  

    
    
