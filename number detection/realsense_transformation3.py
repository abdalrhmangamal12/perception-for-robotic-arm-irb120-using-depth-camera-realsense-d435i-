#!/usr/bin/env python3
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from geometry_msgs.msg import Point
import tf2_ros
import tf2_geometry_msgs
from python_test.srv import RealSense, RealSenseRequest, RealSenseResponse

from abb_robot_msgs.srv import SetIOSignal, SetIOSignalRequest, SetIOSignalResponse
import numpy as np

import tf
from tf import transformations as tr
import ast
class realsense(object):

    def __init__(self):
        self.node=rospy.init_node("rs_rs", anonymous=True)
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1))
        #self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.listener = tf.TransformListener()
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        self.move_group.set_max_acceleration_scaling_factor(0.02) #0.02
        self.move_group.set_max_velocity_scaling_factor(1)
        self.service = rospy.ServiceProxy('number_service', RealSense)

        self.target_link = 'world'
        self.source_link = 'camera_color_optical_frame'
        # rospy.wait_for_service('/rws/set_io_signal')
        # self.proxy = rospy.ServiceProxy('/rws/set_io_signal', SetIOSignal)

        # Links of end effector are: 
        # 1- tool0_gripper_inner
        # 2- tool0_gripper_outer
        # 3- tool0_magnetic_tip
        # 4- tool0_suction_cup

        self.referred_link = "camera_link"


        # self.quaternions = tf.quaternion_from_euler(0,-np.pi/2,np.pi)
        #self.quaternions = tf.quaternion_from_euler(np.pi/2,-np.pi/2,0)
        # self.quaternions = tf.quaternion_from_euler(np.pi/2,0,np.pi/2)






    def begin_to_detect(self):

    #    self.move_group.set_joint_value_target([np.deg2rad(0),np.deg2rad(-45),np.deg2rad(0),np.deg2rad(0),np.deg2rad(110),np.deg2rad(0)])
        print('.....')
       # self.move_group.go(wait=True)
        # self.proxy(signal='valve',value='0')
        self.move_group.set_named_target('begin_to_look')
        self.move_group.go(wait=True)
        rospy.sleep(5)


        point_response = self.service(1)
        point_response=ast.literal_eval(point_response.res)
        all_point=[[key,value] for key,value in point_response.items()]   
        print(all_point)
        self.move_group.set_pose_reference_frame(self.target_link)
        self.move_group.set_end_effector_link(self.referred_link)

        #camera_to_base = self.tf_buffer.lookup_transform(self.source_link, self.target_link , rospy.Time(0))


        self.listener.waitForTransform(self.target_link, self.source_link, rospy.Time(), rospy.Duration())
        trans, rot = self.listener.lookupTransform(self.target_link, self.source_link, rospy.Time(0))

        homo=tr.concatenate_matrices(tr.translation_matrix(trans), tr.quaternion_matrix(rot))
        print("rot t",homo)

        #print("camera_point",camera_to_base)


        camera_point = geometry_msgs.msg.PointStamped()
        camera_point.header.stamp = rospy.Time.now()
        camera_point.header.frame_id = self.source_link
        
        
        

        point= np.array([[all_point[0][1][0]],[all_point[0][1][1]],[all_point[0][1][2]],[1]])
        point_transform= np.dot(homo,point)
       # print(camera_point)

        
        #base_point = self.listener.transformPoint(self.target_link, camera_point)

       # base_point = tf2_geometry_msgs.do_transform_pose(camera_point, camera_to_base)
        #base_point = tf2_geometry_msgs.do_transform_point(camera_point,camera_to_base)


        
        
        try:
           
            new_eef_pose = geometry_msgs.msg.Pose()
            # new_eef_pose.orientation =  copy.deepcopy(base_point.pose.orientation)
            quats = tr.quaternion_from_euler(0,np.pi/2,0)

            new_eef_pose.orientation.x = quats[0]
            new_eef_pose.orientation.y = quats[1]
            new_eef_pose.orientation.z = quats[2]
            new_eef_pose.orientation.w = quats[3]


            new_eef_pose.position.x = point_transform[0][0]
            new_eef_pose.position.y =  point_transform[1][0]
            new_eef_pose.position.z =  point_transform[2][0]

            new_eef_pose.position.z += 0.04

            print(new_eef_pose)
            self.move_group.set_pose_target(new_eef_pose)

            #self.move_group.set_position_target([new_eef_pose.position.x,new_eef_pose.position.y,new_eef_pose.position.z])

            success = self.move_group.go(wait=True)




        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("-----------------\nEXCEPTION\n-----------------")
        # rospy.loginfo('Pose of the object in the world reference frame is:\n %s', camera_point)
        # rospy.loginfo('Pose of the object in the logical camera reference frame is:\n %s', base_point)


if __name__== '__main__':

    r=realsense()
 #   try:
    while not rospy.is_shutdown():
        input("Enter ")
	       
        r.begin_to_detect()

#    except (rospy.service.ServiceException, rospy.ROSInterruptException):
#        moveit_commander.roscpp_shutdown()
#        print('stop')
