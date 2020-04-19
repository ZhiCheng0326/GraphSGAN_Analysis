# To construct grid map....
## 1. Change robot env variable to Hokuyo
```
TURTLEBOT_3D_SENSOR=AstraHokuyo
echo $TURTLEBOT_3D_SENSOR         #check if  launch successfully
```
## 2. Launch Hokuyo robot
Note: Change world file dir

```
roslaunch robot_hokuyo_astra my_world_hk.launch world_file:=/home/zlee/catkin_ws/src/turtlebot_custom_gazebo_worlds/hw2.world
```
## 3. Launch gmapping_demo.launch
Note: change the robot env variable in new terminal before launch gmapping
```
roslaunch robot_hokuyo_astra gmapping_demo.launch
```
## 4. Launch RVIZ
```
roslaunch turtlebot_rviz_launchers view_navigation.launch
```
Note: Change global & local costmap topic to "/map"

## 5. Launch keyboard_teleop
```
roslaunch turtlebot_teleop keyboard_teleop.launch
```
## 6. Save the grid map result
```
rosrun map_server map_saver -f /home/zlee/catkin_ws/src/turtlebot_learn/maps/E1
```

# Grid map get! Let's use it for navigation
## 1. Launch turtlebot
Note: Change world dir
```
roslaunch turtlebot_gazebo turtlebot_world.launch world_file:=/home/zlee/catkin_ws/src/turtlebot_custom_gazebo_worlds/hw2_v2.world
```
## 2. Launch amcl_demo with turtlebot
```
roslaunch turtlebot_gazebo amcl_demo.launch map_file:=/home/zlee/catkin_ws/src/turtlebot_learn/maps/E1.yaml
```
## 3. Launch RVIZ
```
roslaunch turtlebot_rviz_launchers view_navigation.launch
```
## 4. Navigate it to desire place
```
python go_to_specific_point_on_map.py
```



# Extra notes
## Launch amcl with robot hokuyo
```
roslaunch robot_hokuyo_astra amcl_demo.launch map_file:=/home/zlee/catkin_ws/src/turtlebot_learn/maps/E1.yaml

```
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
