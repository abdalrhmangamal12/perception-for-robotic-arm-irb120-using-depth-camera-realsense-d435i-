# Perception Module for Gearbox Part Detection

The perception module is responsible for the detection and identification of gearbox parts using computer vision techniques.
## System Requirements
 * ROS Noetic
*  Ubuntu 20.04
## Setup Instructions
1. Installing Dependencies
* realsense d435i SDK 2.0 instalation yuo can follow documentation of realsense  [link](https://dev.intelrealsense.com/docs/compiling-librealsense-for-linux-ubuntu-guide)

* install cuda and cuddn  [link of documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
  
* clone the repo and run it 
```bash 
cd ~/catkin_ws/src/
git clone git@github.com:abdalrhmangamal12/perception-for-robotic-arm-irb120-using-depth-camera-realsense-d435i-.git 
cd ~/catkin_ws/
catkin_make
cd mechanical parts detection/
python3 full_init.py 
```

## methodology and steps 
### Camera Selection

We used the Intel RealSense D435i depth camera for capturing 3D images Here's why we chose the D435i:

- Offers quality depth for various applications.
- Wide field of view suitable for robotics and augmented reality.
- Range up to 10m, making it versatile.
- Customizable software for flexibility.
- Cost-effective solution.
- Lightweight and powerful.
  
  ![d435i](https://http2.mlstatic.com/D_NQ_NP_604132-MLB69664806665_052023-O.webp)

### 1.caliberation 
#### Dynamic Calibration

Dynamic calibration involves rectification and depth scale calibration. Here's a brief overview:

#### Rectification Calibration

- Aligns epipolar lines for correct depth pipeline.
- Reduces holes in depth image.

#### Depth Scale Calibration

- Aligns depth frame due to optical element changes.

#### Calibration Steps

1. Capture real-time images from left (L) and right (R) cameras.
2. Extract features from images.
3. Match features between L and R cameras.
4. Bin the image into a 6x8 grid and check for corresponding points.
5. Iteratively adjust until all bins have enough features.
6. Optimize extrinsic parameters to minimize rectification error.
   
![dynamic](./mechanical%20parts%20detection/photos/dynamic%20caliberation%20.png)

#### Hybrid Calibration

Hybrid calibration combines target-less and targeted calibration for convenience and accuracy. It rectifies left/right images and performs targeted scale calibration.

![dynamic](./mechanical%20parts%20detection/photos/hybrid.jpeg)

#### On-Chip Self Calibration

Self-calibration methodology for improving depth perception.

#### Quality Depth Measurement

We used the Quality Depth tool to measure depth quality. The RMS error is within an allowable range for various scenarios.

![dynamic](./mechanical%20parts%20detection/photos/on%20ship.jpeg)

### Data Collection and Augmentation

We started collecting data from capturing photos of the 3d printed parts to make our dataset.

We have made augmentation to the data we have by the following augmentations.
* Flip: Horizontal, Vertical
* 90° Rotate: Clockwise, Counter-Clockwise, Upside Down
* Rotation: Between -15° and +15°
* Brightness: Between -25% and +25%
* Exposure: Between -25% and +25%
* nd split it using Roboflow to:
    1.Model training.
    2.Model validation.
* **labeling data up to 10k image using roboflow platform** 
[check project in roboflow platform ](https://universe.roboflow.com/abdalrahman-gamal/mechanical-parts-detection-glf33)

### Model Training

* using yolo-7 object detector to train the model 
* Model Improvement and tune hyperprameters 

### Circle Detection

The problem is that the centre of the bounding box that surround the gear is not the required centre point that the robot must reach to it, the real centre of the gear is the centre point of the circle that is why we needed to implement an algorithm to determine the internal hole of the gear.

![circle](./mechanical%20parts%20detection/photos/circle.jpeg)

### Line Detection for Bolt Orientation Estimation

Using the Hough Transform for bolt orientation estimation.

![line](./mechanical%20parts%20detection/photos/line.png)

### Pixel to 3-D Point Alignment & Point Cloud

Mapping 2D pixels to 3D points in the point cloud.

```python 
  def pointconvert(self,depth_intrin,numb_dictionary,depth_frame):
   
  '''get 2d pixel and get the 3d postion with recpect to camera frame'''
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
```

## Eye-in-Hand Calibration

Calibrating the camera mounted on the robot's end effector.

![hand in eye ](./mechanical%20parts%20detection/photos/hand.jpeg)

## Autoexposure Filter

Improving image quality with autoexposure.
**before using filter** 

![line](./mechanical%20parts%20detection/photos/1.jpeg)

**after using filter**

![line](./mechanical%20parts%20detection/photos/2.jpeg)

## Number Detection

The number detection algorithm is used in our   application to recognize the number of each grid. In our application, there are 4 grids each with its number, this number has an indication about the parts in its grid.


## Results 
watch youtube video

[![Watch the Video](https://img.youtube.com/vi/ZXA9SAabvSY/0.jpg)](https://www.youtube.com/watch?v=ZXA9SAabvSY)


Here we are comparing our precision-recall curves with previous models, we could notice that our mechanical parts (classes) are more than the previous model mentioned in section Also, the Recall value of all classes is 92.6%, while the best previous model shows that the recall value was 91% at the same mAP value (0.5).

![related work](./mechanical%20parts%20detection/photos/related_work.jpeg)



![recall](./mechanical%20parts%20detection/photos/recall.jpeg)


