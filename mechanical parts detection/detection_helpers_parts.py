#!/usr/bin/env python
import argparse
import time
from pathlib import Path
import rospy
from sensor_msgs.msg import Image
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
#from python_test.srv import RealSense, RealSenseRequest, RealSenseResponse
import rospy
from sensor_msgs.msg import Image
from ros_numpy import msgify


gears_informdict={} 




class parts_Detector:
    def __init__(self):

        self.weights = '/home/abdo_gamal/gp_ws/src/python_test/scripts/yolov7-new/yolov7/runs/train/exp/weights/best.pt'
        self.source = 'inference/images'
        self.img_size = 640
        self.conf_thres = 0.45
        self.iou_thres = 0.5
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
            dx -=0.008
            dy -= 0.004
            print("depth from point to camera ", depth)
            print(f'x is {dx} and y is {dy} and z is {dz}')
            gears_with_depth_point.append([dx,dy,dz,i[2]])
        print('point with theta',gears_with_depth_point)
        numb_inform=dict(zip(numb_dictionary.keys(),gears_with_depth_point))
        return numb_inform
    
    def detect_circle(self,img):

        circle_det_center=[]

    #'method 2' ..
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #    circle detection using hough transform
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,30,param1=50,param2=50,minRadius=0,maxRadius=0)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,30,param1=45,param2=45,minRadius=0,maxRadius=0)
        if circles is not None:
           # print('in circle detection')
            circles = np.round(circles[0, :]).astype("int")
            for (x,y,r) in circles:
                        
                cv2.circle(img, (x,y), r, (36,255,12), 3)  
                circle_det_center.append([x,y])   
                    
        #                       #end of edge detection
      #  cv2.circle(im0,(round(circle_det_center[0][0]),round(circle_det_center[0][1])),3,(0,255,255),2)
        #print('ddfgfgfhgh',circle_det_center)
        return circle_det_center
    

    def line_detection(self,image):

        gray_blur = cv2.GaussianBlur(image, (5, 5), 0)
            # Detect edges using Canny edge detection
        edges = cv2.Canny(gray_blur, 5, 50)
        # Apply Hough line transform
     #   cv2.imshow('edges',edges)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 30)
        print('dsgdfhgfhjkjjkhjkjohj',lines)
        angales =[]
        if lines is not None:
            for line in lines:
                    rho, theta = line[0]
                   # print(theta)
                    ang=round((theta*(180/np.pi)))
                   
                   # angales.append(ang>0)
                    #print("ang",ang)
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho


                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    if ang!=0 and ang!=90:
                         angales.append(ang)
                    
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
      #  print('anagles',angales) 
        cv2.namedWindow('line detection', cv2.WINDOW_NORMAL)
       
        cv2.imshow('line detection',image)           
        theta = max(angales,key=lambda x: angales.count(x))
        #print('theta',theta)
        return theta


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
            circle_det_center=[]
            m=vertics_rec_array.shape
            for i in range(0,m[0]):
                 x_circle=(((vertics_rec_array[i][0])+(vertics_rec_array[i][2]))/2)
                 y_circle=(((vertics_rec_array[i][1])+(vertics_rec_array[i][3]))/2)
                 theta =0
                 cv2.circle(im0,(round(x_circle),round(y_circle)),3,(0,255,0),2)
                 if self.labels[i] == 'gear' or self.labels[i] == 'bearing' :
                     circle_det_center=self.detect_circle(im0[round(vertics_rec_array[i][1]):round(vertics_rec_array[i][3]),round(vertics_rec_array[i][0]):round(vertics_rec_array[i][2])])
               #  if circle_det_center is not None:
                #     cv2.circle(im0,(round(circle_det_center[0][0]),round(circle_det_center[0][1])),3,(0,255,255),2)
                  #   x_circle ,y_circle,theta= circle_det_center[0][0],circle_det_center[0][1],0

                 try:
                 	if  self.labels[i] == 'bolt' :
                 		theta=self.line_detection(im0[round(vertics_rec_array[i][1]):round(vertics_rec_array[i][3]),round(vertics_rec_array[i][0]):round(vertics_rec_array[i][2])])
                 		
                 except:
                 	pass

                 centre_circle.append((round(x_circle),round(y_circle),theta))
     
            gears_inform=dict(zip(self.labels,centre_circle))
               


                        
           # gears_informarray=np.array(list(gears_inform.items()))
        #     # convert 2d point to 3d 3d from convert file using  function pointconvert
            global gears_informdict
            gears_informdict= self.pointconvert(depth_intrin , gears_inform,depth_frame)
          #  print(gears_informdict)
      #      image_msg = msgify(Image, im0, encoding='rgb8')
       #     rate = rospy.Rate(1)
      #      pub = rospy.Publisher('parts_topic', Image, queue_size=10)
      #      pub.publish(image_msg)
     #       rate.sleep()
            cv2.imshow("parts_detection", im0)
           # cv2.imshow(" result", depth_frame)
    
            # cv2.imshow("Recognition result depth",depth_colormap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            return gears_informdict

    
    
