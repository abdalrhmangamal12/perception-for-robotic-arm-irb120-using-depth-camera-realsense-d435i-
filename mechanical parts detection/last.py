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
#from python_test.srv import RealSense, RealSenseRequest, RealSenseResponse
   

gears_informdict={} 
depth_frame = 0


# Configure depth and color streams

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile=pipeline.start(config)

#sensor = profile.get_device().query_sensors()[0]
#sensor.set_option(rs.option.enable_auto_exposure, True)

# Increase gain for high light conditions
#sensor.set_option(rs.option.gain, -3)


align = rs.align(rs.stream.color)

frames = pipeline.wait_for_frames()
frames = pipeline.wait_for_frames()
frames = pipeline.wait_for_frames()


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    vertics_rec=[]
    labels=[]
    

    while(True):
        vertics_rec.clear() 
        labels.clear() 
    # Read a frame from the video stream
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        profile = aligned_frames.get_profile()


        color_frame = aligned_frames.get_color_frame()
        global depth_frame
        depth_frame = aligned_frames.get_depth_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()
     #   print(depth_intrinsics)
        spatial = rs.spatial_filter()
        #spatial.set_option(rs.option.filter_magnitude, 5)
        #spatial.set_option(rs.option.filter_smooth_alpha, 1)
        #spatial.set_option(rs.option.filter_smooth_delta, 50)
        
        filtered_depth = spatial.process(depth_frame)

        img = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(filtered_depth.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)



        # Letterbox
        im0 = img.copy()
        img = img[np.newaxis, :, :, :]        
 
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)


        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
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
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    #plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)
                    c=torch.tensor(xyxy)
                  
                    labels.append(str(label)[0:-5])
                    vertics_rec.append(c.numpy())
            vertics_rec_array=np.array(vertics_rec)
          #  print(vertics_rec_array)
           # print(np.shape(vertics_rec_array))
            centre_circle=[]

            m=vertics_rec_array.shape
            for i in range(0,m[0]):
                 x_circle=(((vertics_rec_array[i][0])+(vertics_rec_array[i][2]))/2)
                 y_circle=(((vertics_rec_array[i][1])+(vertics_rec_array[i][3]))/2)
                 centre_circle.append((round(x_circle),round(y_circle)))
                 cv2.circle(im0,(round(x_circle),round(y_circle)),3,(0,255,0),2)

            gears_inform=dict(zip(labels,centre_circle))
               


                        
           # gears_informarray=np.array(list(gears_inform.items()))
        #     # convert 2d point to 3d 3d from convert file using  function pointconvert
            global gears_informdict
            gears_informdict= pointconvert(depth_intrin , gears_inform)
            print(gears_informdict)
            
            cv2.imshow("Recognition result", im0)
           # cv2.imshow(" result", depth_frame)
    
            # cv2.imshow("Recognition result depth",depth_colormap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def pointconvert(depth_intrin,numb_dictionary):
    
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
       


   


def detect_callback(data):
    response=RealSenseResponse()
    
    for l,i in gears_informdict.items(): 
            print(l)
            
            response.location.header.frame_id='l'
            response.location.pose.position.x=i[0]-0.009
            response.location.pose.position.y=i[1]+0.005
            response.location.pose.position.z=i[2]
    return response


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/abdo_gamal/gp_ws/src/python_test/scripts/yolov7-new/yolov7/runs/train/exp/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    rospy.init_node('node_1', anonymous=False)
    global x_prev , y_prev
    x_prev=0
    y_prev=0

    rospy.Service('realsense_service',RealSense,detect_callback)



    #check_requirements(exclude=('pycocotools', 'thop'))
    try:
       with torch.no_grad():

        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['/home/abdo_gamal/gp_ws/src/python_test/scripts/yolov7-new/yolov7/runs/train/exp/weights/best.pt']:

                detect()
                strip_optimizer(opt.weights)
        else:

            detect()
    except rospy.ROSInterruptException:
        pass
   
