# To construct grid map....
Note: Don't turn and go forward together, press k to break before change from turn to go forward
## 1. Change robot env variable to Hokuyo
```
TURTLEBOT_3D_SENSOR=AstraHokuyo
echo $TURTLEBOT_3D_SENSOR         #check if  launch successfully
```
## 2. Launch Hokuyo robot + world_file
Note: Change world file dir

```
roslaunch robot_hokuyo_astra my_world_hk.launch world_file:=/home/zlee/catkin_ws/src/turtlebot_custom_gazebo_worlds/exam_sim.world
```
## 3. Launch gmapping_demo.launch
Change robot env variable to robot_hokuyo_astra in new terminal before launch gmapping
```
roslaunch robot_hokuyo_astra gmapping_demo.launch
```
## 4. Launch RVIZ
Change global & local costmap topic to "/map"
```
roslaunch turtlebot_rviz_launchers view_navigation.launch
```

## 5. Launch keyboard_teleop
```
roslaunch turtlebot_teleop keyboard_teleop.launch
```
## 6. Save the grid map result
Change directory / file name
```
rosrun map_server map_saver -f /home/zlee/catkin_ws/src/turtlebot_learn/maps/E1
```

# Grid map get! Let's use it for navigation
## 1. Launch robot + world_file
### 1.1 To launch turtlebot
Change world dir
```
roslaunch turtlebot_gazebo turtlebot_world.launch world_file:=/home/zlee/catkin_ws/src/turtlebot_custom_gazebo_worlds/exam_sim.world
```
### 1.2 To launch robot_hokuyo_astra
Change robot env variable to robot_hokuyo_astra
```
roslaunch robot_hokuyo_astra my_world_hk.launch world_file:=/home/zlee/catkin_ws/src/turtlebot_custom_gazebo_worlds/exam_sim.world
```

## 2. Launch amcl_demo with robot
### 2.1 Launch amcl_demo with turtlebot
```
roslaunch turtlebot_gazebo amcl_demo.launch map_file:=/home/zlee/catkin_ws/src/turtlebot_learn/maps/E1.yaml
```
### 2.2 Launch amcl_demo with robot_hokuyo_astra
Don't need to change the robot environment variable
```
roslaunch robot_hokuyo_astra amcl_demo.launch map_file:=/home/zlee/catkin_ws/src/turtlebot_learn/maps/E1.yaml
```
## 3. Launch RVIZ
```
roslaunch turtlebot_rviz_launchers view_navigation.launch
```
## 4. Navigate the robot
### 4.1 In homework
```
python go_to_specific_point_on_map.py
```
### 4.2 In exam
```
python follow_left_wall_find_target.py
python follow_right_wall_find_target.py
```

# Fit object center with robot camera
Prerequisite: Color cube is obtained
## 1. Launch color recognition script
```
python green_patch_recognition_ori.py
```
## 2. Launch fit center script
```
python fit_object_center.py
```


# Extra notes
## Empty world file dir
```
world_file:=/usr/share/gazebo-7/worlds/empty.world
```

## Launch turtlebot in greencube world
```
roslaunch turtlebot_gazebo turtlebot_world.launch world_file:=/home/zlee/catkin_ws/src/turtlebot_custom_gazebo_worlds/green_cube1.world
```
## Open camera
```
rqt_image_view
```
### Get current coordinate of turtlebot
```
rosrun tf tf_echo /map /base_footprint
```
###Shutdown ROS
```
def myhook():
  print "shutdown time!"

rospy.on_shutdown(myhook)

```
## To launch 3 turtlebots...
roslaunch multi_robot main.launch world_file:=/home/zlee/catkin_ws/src/turtlebot_custom_gazebo_worlds/hw2_v2.world
