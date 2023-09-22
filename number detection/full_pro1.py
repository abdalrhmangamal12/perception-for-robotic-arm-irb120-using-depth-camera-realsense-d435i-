#!/usr/bin/env python3

import sys
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
import importlib.util
spec = importlib.util.spec_from_file_location("rospy", "/opt/ros/noetic/lib/python3/dist-packages/rospy/__init__.py")
foo = importlib.util.module_from_spec(spec)
sys.modules["rospy"] = foo
spec.loader.exec_module(foo)
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import actionlib
import geometry_msgs.msg
import std_msgs
from abb_robot_msgs.srv import SetIOSignal, SetIOSignalRequest, SetIOSignalResponse
from python_test.srv import RealSense, RealSenseRequest, RealSenseResponse

from moveit_msgs.msg import PlanningOptions
from moveit_msgs.msg import MoveGroupAction, MoveGroupGoal, PlanningOptions
import numpy as np
import tf2_ros
import tf2_geometry_msgs

import tf.transformations as tf
from numpy import deg2rad as dr
import ast

p_values = [[dr(53),dr(-16),dr(20),dr(0),dr(86),dr(53)]
,[dr(25),dr(32),dr(-34),dr(0),dr(92),dr(25)]
,[dr(-25),dr(32),dr(-34),dr(0),dr(92),dr(-25)]
,[dr(-53),dr(-16),dr(20),dr(0),dr(86),dr(-53)]]

# mmmmm = [dr(0),dr(-45),dr(0),dr(0),dr(110),dr(0)]

poses = [[-0.05,0,0.2,0,0,0,1],[0,-0.1,0.3,0,0,0,1],[0.1,-0.1,0.2,0,0,0,1],[0.1,0,0.2,0,0,0,1]]

class fullPro(object):
    def __init__(self):
        rospy.init_node('tracker',anonymous=False)

        # First initialize moveit_commander and rospy.
        moveit_commander.roscpp_initialize(sys.argv)


        # Instantiate a MoveGroupCommander object.  This object is an interface
        # to one group of joints.  In this case the group refers to the joints of
        # robot1. This interface can be used to plan and execute motions on robot1.

        self.move_group = moveit_commander.MoveGroupCommander("manipulator")
        self.scene = moveit_commander.PlanningSceneInterface()
        self.robot = moveit_commander.RobotCommander()

        
        self.move_group.set_max_acceleration_scaling_factor(0.02)
        self.move_group.set_max_velocity_scaling_factor(1)
        rospy.loginfo('Execute Trajectory server is available for robot1')
        self.move_group.set_planning_time(0.2)
        # robot1_group.set_num_planning_attempts(1)
        self.move_group.allow_replanning(True)  # allow for replanning if obstacles are encountered
        self.move_group.allow_looking(True)

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.robot_client = actionlib.SimpleActionClient(
        'execute_trajectory',
        moveit_msgs.msg.ExecuteTrajectoryAction)
        self.robot_goal = moveit_msgs.msg.ExecuteTrajectoryGoal()

        self.target_link = 'world'
        self.source_link = 'camera_color_optical_frame'
        self.move_group.set_pose_reference_frame(self.target_link)


        # Links of end effector are: 
        # 1- tool0_gripper_inner
        # 2- tool0_gripper_outer
        # 3- tool0_magnetic_tip
        # 4- tool0_suction_cup

        self.referred_link = 'tool0_gripper_outer'
        self.move_group.set_end_effector_link(self.referred_link)




        self.quaternions = tf.quaternion_from_euler(0,0,0) # camera optical frame to any end effector frame
        # self.quaternions = tf.quaternion_from_euler(0,0,0) # base_link to any end effector frame
        self.position = geometry_msgs.msg.PoseStamped()



        self.pose_point = self.move_group.get_current_joint_values()
        rospy.Subscriber('pose_estimation',geometry_msgs.msg.PoseStamped,self.hand_track_callback,queue_size=1)

        self.p = rospy.Publisher('camera_list',std_msgs.msg.String,queue_size=1)
        self.p_2 = rospy.Publisher('output',std_msgs.msg.String,queue_size=1)
        self.p_3 = rospy.Publisher('pose_flag',std_msgs.msg.Int16,queue_size=1)

        self.number_service = rospy.ServiceProxy('number_service', RealSense)
        self.parts_service = rospy.ServiceProxy('parts_service', RealSense)


        # self.get_list()
        self._list_callback([['1','gear1','gear','8'],['1','gear1','gear','8'],['1','gear1','gear','8'],['2','bolt1','bolt','6'],['3','nut1','nut','6'],['1','gear1','bearing','9']])




    def get_list(self):
        rospy.Subscriber('list_file_analyzed',std_msgs.msg.String,self._list_callback,queue_size=1)
        rospy.spin()
   

    def _list_callback(self,list_analyzed):

        done = False

        if done == True:
            print("done getting parts")

        else:
            analyzed_parts  = list_analyzed #[inner_string.split(', ') for inner_string in list_analyzed.data.split('\n')]
            
            rospy.wait_for_service('/number_service')
            rospy.wait_for_service('/parts_service')
            self.survey_for_grid_points()
            
            self.survey_for_parts()

            print(1)

            self.available_analyzed_parts = []
            self.not_available_analyzed_parts = []

            # analyzed_parts = [[1,name,type,size],[]] 




            available_parts = [any(element[2:] == item[2:] for item in self.camera_list) for element in analyzed_parts]
            self.available_analyzed_parts = [item for item, is_available in zip(analyzed_parts, available_parts) if is_available]
            self.not_available_analyzed_parts = [item for item, is_available in zip(analyzed_parts, available_parts) if not is_available]



            print(self.available_analyzed_parts,self.not_available_analyzed_parts)
            list_1 = '\n'.join([' '.join(map(str, sublist)) for sublist in self.camera_list])
            list_2 = "\n\nAvailable Parts:\n"+'\n'.join([' '.join(map(str, sublist)) for sublist in self.available_analyzed_parts])
            list_3 = "\n\nNot Available Parts:\n"+'\n'.join([' '.join(map(str, sublist)) for sublist in self.not_available_analyzed_parts])

            print(list_1+list_2+list_3)
            self.p.publish(list_1+list_2+list_3)



            # self.go_to_location(predefined_location='begin_to_look')



            for sublist in self.available_analyzed_parts:
                # name = 'Nut', size = '10'
                self.move_group.stop()
                self.move_group.clear_pose_targets()
                
                name,size = sublist[2:]
                print(f'iteration: {name,size}')
                
                grid_point = next(item[1] for item in self.grid_points if item[0] == size)
                print(grid_point)

                self.target_link = 'world'
                self.source_link = 'world'
                self.referred_link = 'camera_link'
                self.quaternions = tf.quaternion_from_euler(0,np.pi/2,0)
                self.move_group.set_end_effector_link(self.referred_link)

                self.move_group.set_pose_target(grid_point)
                
                success = False
                while not success:

                	success = self.move_group.go(wait=True)
                self.move_group.stop()
                self.move_group.clear_pose_targets()
                rospy.sleep(3)
                  


                returned_ = self.parts_service(1) # [[8,point_location],[12,point_location]]
                print(returned_)
                returned_dict = ast.literal_eval(returned_.res)
                returned_list = [[key, value] for key, value in returned_dict.items()]

                print(returned_list)

                # data = [['8', [3, 2, 1]], ['9', [4, 5, 6]]]

                result = [part_position[1] for part_position in returned_list if part_position[0] == name]
                if result != None:
                    print(result)


                    # delete later
                    part_position = geometry_msgs.msg.PoseStamped()
                    part_position.pose.position.x = result[0][0]
                    part_position.pose.position.y = result[0][1]
                    part_position.pose.position.z = result[0][2]


                    self.target_link = 'world'
                    self.source_link = 'camera_color_optical_frame'
                    self.quaternions = tf.quaternion_from_euler(0,np.pi/2,0)

                    if name == 'gear':
                        self.referred_link = 'tool0_gripper_inner'

                    elif name == 'nut' :
                        self.referred_link = 'tool0_magnetic_tip'

                    elif name == 'bearing':
                        self.referred_link = 'tool0_gripper_inner'
                    else:
                        self.referred_link = 'tool0_gripper_outer'

                    self.get_part(part_position)
                    rospy.sleep(1)
                    self.p_2.publish(f'{name} {size} is sent.')
                
                else:
                    self.p_2.publish(f'{name} {size} already been taken.')


            done = True





    def survey_for_parts(self):
        self.camera_list = []

        self.target_link = 'world'
        self.source_link = 'world'
        self.referred_link = 'camera_link'
        self.quaternions = tf.quaternion_from_euler(0,np.pi/2,0)



        for i in range(0, len(self.grid_points)):
            index = 1
            print(self.grid_points[i][1])



            self.grid_points[i][1].pose.orientation.x=self.quaternions[0]
            self.grid_points[i][1].pose.orientation.y=self.quaternions[1]
            self.grid_points[i][1].pose.orientation.z=self.quaternions[2]
            self.grid_points[i][1].pose.orientation.w=self.quaternions[3]



            self.grid_points[i][1].pose.position.x += -0.08
            self.grid_points[i][1].pose.position.y +=  0.07
            self.grid_points[i][1].pose.position.z +=  0.30


            self.move_group.set_pose_target(self.grid_points[i][1])
            
            
            success = False
            while not success:
            	success = self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()

            rospy.sleep(3)
            
            returned_ = self.parts_service(1)

            returned_dict = ast.literal_eval(returned_.res)
            returned_list = [key for key in returned_dict.keys()]

                
            for element in returned_list:
                self.camera_list.append([index, element[0] + str(index), element, self.grid_points[i][0]])
                index += 1






        # self.camera_list = [[1,name,type,size],[]]

    def survey_for_grid_points(self):
        print(2)
        self.grid_points = []
        
        self.target_link = 'world'
        self.source_link = 'camera_color_optical_frame'
        self.referred_link = 'camera_link'
        self.quaternions = tf.quaternion_from_euler(0,np.pi/2,0)
        

        self.move_group.set_joint_value_target([0,dr(-45),0,0,dr(110),0])
        success = False
        while not success:
        	success = self.move_group.go(wait=True)

        self.move_group.stop()
        self.move_group.clear_pose_targets()
        
        self.move_group.set_pose_reference_frame(self.target_link)
        self.move_group.set_end_effector_link(self.referred_link)

        camera_to_base = self.tf_buffer.lookup_transform(self.target_link, self.source_link , rospy.Time(0))

        camera_point = geometry_msgs.msg.PoseStamped()
        camera_point.header.frame_id = self.source_link
        print(3)

        rospy.sleep(3)
        print(4)

        returned_ = self.number_service(1) # [[8,point_location],[12,point_location]]
        returned_dict = ast.literal_eval(returned_.res)
        returned_list = [[key, value] for key, value in returned_dict.items()] 

        for element in returned_list:
            

            camera_point.header.stamp = rospy.Time.now()

            camera_point.pose.position.x = element[1][0]
            camera_point.pose.position.y = element[1][1]
            camera_point.pose.position.z = element[1][2]

            base_point = tf2_geometry_msgs.do_transform_pose(camera_point, camera_to_base)

            self.grid_points.append([element[0],base_point])




    def go_to_location(self,position = False,joint_values = False ,predefined_location = False):
        
        success = False
        choice = 0
        
        while not success:


            if choice == 0:
                if position:



                    camera_to_base = self.tf_buffer.lookup_transform(self.target_link, self.source_link, rospy.Time(0))

                    camera_point = geometry_msgs.msg.PoseStamped()
                    camera_point.header.stamp = rospy.Time.now()
                    camera_point.header.frame_id = self.source_link

                    

                    camera_point.pose.position.x = position.pose.position.x
                    camera_point.pose.position.y = position.pose.position.y
                    camera_point.pose.position.z = position.pose.position.z

                    base_point = tf2_geometry_msgs.do_transform_pose(camera_point, camera_to_base)
                    print(base_point)

                    base_point.pose.orientation.x = self.quaternions[0]
                    base_point.pose.orientation.y = self.quaternions[1]
                    base_point.pose.orientation.z = self.quaternions[2]
                    base_point.pose.orientation.w = self.quaternions[3]


                    self.move_group.set_pose_target(base_point)
                    print(base_point)
                    choice += 1
                
                elif joint_values:

                    self.move_group.set_joint_value_target(joint_values)
                    choice += 1

                elif predefined_location:
                    self.move_group.set_named_target(predefined_location)
                    choice += 1


            current_state = np.asarray(self.move_group.get_current_joint_values())
            success=self.move_group.go(wait=True)

                

            current_time = rospy.Time.now()
            while np.linalg.norm(current_state - np.asarray(self.move_group.get_current_joint_values()))<0.001:
                
                self.move_group.go(wait=False)
                rospy.sleep(1)
                print(f'looped time: {np.abs((current_time - rospy.Time.now()).to_sec())}',end='\r')

                if np.abs((current_time - rospy.Time.now()).to_sec()) > 15:
                    success = True
                    print("EXITING, TIME LIMIT REACHED")
                    
                    break
                



    def get_part(self,point_response):
        self.move_group.set_planning_pipeline_id('ompl')
        self.move_group.set_planner_id('RRTConnect')
        self.target_link = 'world'
        #self.quaternions=tf.quaternions_from_euler(0,np.pi/2,0)

        
        self.move_group.set_pose_reference_frame(self.target_link)

        self.move_group.set_end_effector_link(self.referred_link)

        camera_to_base = self.tf_buffer.lookup_transform(self.target_link, self.source_link, rospy.Time(0))

        camera_point = geometry_msgs.msg.PoseStamped()
        camera_point.header.stamp = rospy.Time.now()
        camera_point.header.frame_id = self.source_link



            

        camera_point.header.stamp = rospy.Time.now()

        camera_point.pose.position.x = point_response.pose.position.x
        camera_point.pose.position.y = point_response.pose.position.y
        camera_point.pose.position.z = point_response.pose.position.z
            
        base_point = tf2_geometry_msgs.do_transform_pose(camera_point, camera_to_base)
        print(base_point)

        try:

            new_eef_pose = geometry_msgs.msg.Pose()
            new_eef_pose.orientation.x = self.quaternions[0]
            new_eef_pose.orientation.y = self.quaternions[1]
            new_eef_pose.orientation.z = self.quaternions[2]
            new_eef_pose.orientation.w = self.quaternions[3]
            
            new_eef_pose.position =  copy.deepcopy(base_point.pose.position)
            new_eef_pose.position.z+=0.15
            print(new_eef_pose)
            
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
#            p = geometry_msgs.msg.PoseStamped()
            
#            p.pose.position.x = 0.34357415454368934
#            p.pose.position.y = 0.14201443362051594
#            p.pose.position.z = 1.1189679103585601
            
            self.move_group.set_pose_target(new_eef_pose)
            
            success = False
            while not success:
 	           success = self.move_group.go(wait=True)
 	           
 	    
            rospy.sleep(1)


            self.move_group.stop()
            self.move_group.clear_pose_targets()

               # base_point.pose.position.z+=0.05
            new_eef_pose.position =  copy.deepcopy(base_point.pose.position)
            if self.referred_link == 'tool0_magnetic_tip':
                   new_eef_pose.position.z += 0.005
 
            elif self.referred_link == 'tool0_gripper_outer':
                   new_eef_pose.position.z += 0.0
                   self.move_group.set_planning_pipeline_id('pilz_industrial_motion_planner') 

                   self.move_group.set_planner_id('LIN')  
            elif self.referred_link == 'tool0_gripper_inner':
                   new_eef_pose.position.z += 0.007
                   self.move_group.set_planning_pipeline_id('pilz_industrial_motion_planner') 

                   self.move_group.set_planner_id('LIN') 
                                                                         

            
            
                # print(new_eef_pose)

                # x,y --> ompl / z --> pilz (LIN)
                # (ompl, pilz_industrial_motion_planner)
                # (RRTConnect, LIN)


            self.move_group.set_pose_target(new_eef_pose)
            
            success = False
            while not success:
               success = self.move_group.go(wait=True)
 	           
 	           
            if success:
               print("-----------------\nSUCCESS\n-----------------")
               self.move_group.stop()
               self.move_group.clear_pose_targets()
               rospy.sleep(1)
               #self.proxy(signal='valve',value='1')
            
            self.move_group.stop()
            self.move_group.clear_pose_targets()

            self.move_group.set_end_effector_link(self.referred_link)
            new_eef_pose.position =  copy.deepcopy(base_point.pose.position)
            
            new_eef_pose.position.z+=0.15
                
            self.move_group.set_pose_target(new_eef_pose)
            success = False
            while not success:

               success = self.move_group.go(wait=True)
                

            #    self.move_group.set_planning_pipeline_id('ompl')
            #    self.move_group.set_planner_id('RRTConnect')
                

             #   self.move_group.set_named_target('middle_right')
              #  self.move_group.go(wait=True)
              #  rospy.sleep(3)
            self.move_group.set_planning_pipeline_id('ompl')
            self.move_group.set_planner_id('RRTConnect')
            #self.proxy(signal='valve',value='0')


        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("-----------------\nEXCEPTION\n-----------------")
        # rospy.loginfo('Pose of the object in the world reference frame is:\n %s', camera_point)
        # rospy.loginfo('Pose of the object in the logical camera reference frame is:\n %s', base_point)



    def hand_track_callback(self,location):

        camera_to_base = self.tf_buffer.lookup_transform(self.target_link, self.source_link, rospy.Time(0))

        camera_point = geometry_msgs.msg.PoseStamped()
        camera_point.header.stamp = rospy.Time.now()
        camera_point.header.frame_id = self.source_link

        quats = self.tf_buffer.lookup_transform(self.source_link, self.referred_link,rospy.Time(0))

        camera_point.pose.position.x = location.pose.position.x
        camera_point.pose.position.y = location.pose.position.y
        camera_point.pose.position.z = location.pose.position.z


        self.pose_point = tf2_geometry_msgs.do_transform_pose(camera_point, camera_to_base)




    def hand_pose_target(self):

        #delete later
        self.target_link = 'world'
        self.source_link = 'camera_color_optical_frame'
        
        self.move_group.set_max_acceleration_scaling_factor(0.04)
        self.move_group.set_max_velocity_scaling_factor(1)

        reached = False
        # rospy.Subscriber('mouse_location',geometry_msgs.msg.PoseStamped,self._callback,queue_size=1)

        while not reached:
            success = False


            current_location = self.move_group.get_current_pose(self.referred_link)
            
            current_location = np.asarray([current_location.pose.position.x,
            current_location.pose.position.y,
            current_location.pose.position.z])
            

            point_received = np.asarray([self.pose_point.pose.position.x,
            self.pose_point.pose.position.y,
            self.pose_point.pose.position.z])

            current_time = rospy.Time.now()
            
            while np.linalg.norm(current_location-point_received)<0.1:
                
                # rospy.Subscriber('mouse_location',geometry_msgs.msg.PoseStamped,self._callback,queue_size=1)
                
                point_received = np.asarray([self.pose_point.pose.position.x,
                    self.pose_point.pose.position.y,
                    self.pose_point.pose.position.z])

                print(f'looped time: {np.abs((current_time - rospy.Time.now()).to_sec())}',end='\r')

                if np.abs((current_time - rospy.Time.now()).to_sec()) > 4:
                    reached = True
                    print("EXITING, REACHED HAND")
                    break


            if reached:
                break

            else:

                while not success:
                    # print(base_point)
                    # rospy.Subscriber('mouse_location',geometry_msgs.msg.PoseStamped,self._callback,queue_size=1)
                    self.move_group.set_position_target(self.pose_point)
                    current_state = np.asarray(self.move_group.get_current_joint_values())
                    success=self.move_group.go(wait=True)
                    

                    current_time = rospy.Time.now()
                    while np.linalg.norm(current_state - np.asarray(self.move_group.get_current_joint_values()))<0.01:
                        # rospy.Subscriber('mouse_location',geometry_msgs.msg.PoseStamped,self._callback,queue_size=1)
                        self.move_group.set_position_target(self.pose_point)
                        self.move_group.go(wait=False)

                        rospy.sleep(1)
                        print(f'waiting looped time: {np.abs((current_time - rospy.Time.now()).to_sec())}',end='\r')

                        if np.abs((current_time - rospy.Time.now()).to_sec()) > 15:
                            success = True
                            reached = True
                            print("EXITING, TIME LIMIT REACHED")
                            break




if __name__ == '__main__':
    try:
        fullPro()
    except rospy.ROSInterruptException:
        pass
