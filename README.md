# DuelingQnet_with_RosEnv
Implementation of a Dueling Qnetwork Agent on Tensorflow that utilises ROS in order to Interact with the enviroment

**getting started:**
  provided will be a anaconda enviroment.yml
  
  
```python
conda env create -f ros_n_flow.yml
```
or 
```python
source env create -f ros_n_flow.yml
```
the enviroment is setup with nesseracy modules and their correct version

Install [ROS Kinetic](http://wiki.ros.org/kinetic/Installation)

download workspace
  follow [guide](https://answers.ros.org/question/193901/how-to-migrate-a-catkin-workspace/) on migrating a workspace
**finally **
```
roslaunch proto_net agent_n_enviroment.launch 
```

**ELSE** 
the long way:
  * Create [ws](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)
  * Create package(http://wiki.ros.org/ROS/Tutorials/BuildingPackages)
  * Copy srv launch scripts msg into your package 


  * Copy srv launch scripts msg into your package
  * Modify CMakeList.txt to for [srv and msg](http://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv)
  * In workspace call:
```
catkin_make
```
**finally **
```
roslaunch proto_net agent_n_enviroment.launch 
```


