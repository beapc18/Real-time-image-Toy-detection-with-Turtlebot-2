# Toy Detection with Turtlebot 2.
The **goal** of this project is to **detect a specific toy** among a small group of toys scattered around a room. The robot should observe its environment looking for the correct toy among the bunch, then move towards it correctly and, finally, signal that it has been found. The detection must be carried away by a **Convolutional Neural Network** trained with augmented training data.

<p align="center">
<img src="https://github.com/beapc18/MSC_HumanoidRoboticsLab/blob/master/images/platypus.gif" width="80%">  
</p>


Three different neural networks have been trained, one for each of the selected toys, following the [Tensorflow Object Detection API Tutorial](https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/) 

<p align="center">
<img src="https://github.com/beapc18/MSC_HumanoidRoboticsLab/blob/master/images/toy_platypus.jpg" width="30%">
<img src="https://github.com/beapc18/MSC_HumanoidRoboticsLab/blob/master/images/toy_teddy_monster.jpg" width="30%">
<img src="https://github.com/beapc18/MSC_HumanoidRoboticsLab/blob/master/images/toy_unicorn.jpg" width="30%">
</p>

The robot used in this project is the [TurtleBot 2](https://www.turtlebot.com/turtlebot2/), equipped with two cameras, one located in the middle of the robot facing forward, and the other one located on top of the robot, which is tilted slightly downwards to face the ground in front of the robot. The mobile base has three bumpers to detect impacts.

<p align="center">
<img src="https://github.com/beapc18/MSC_HumanoidRoboticsLab/blob/master/images/frontview_components.png" width="30%">  
</p>

We developed two modules for our project, which interact as follows:
<p align="center">
<img src="https://github.com/beapc18/MSC_HumanoidRoboticsLab/blob/master/images/diagram_with_topics.png" width="80%">
</p>

The **detection module** is in charge of the recognition of the different toys and the extraction of positional and proximity information regarding the toy detections. This information is then sent to the movement module, which will steer the robot accordingly.

The main purpose of the **movement module** is to guide the TurtleBot through the environment in order to find the selected toy, based on the messages sent by the detection module.


## Setup guide

### Requirements
- Python 3
- ROS melodic
- TurtleBot 2 (or Gazebo and TurtleBot2 for Gazebo)
- Install python libraries included in requirements.txt
- Python-cv-bridge
- Python-rospy

### Prepare ROS workspace
First of all, you need to initialize your ROS workspace as explained in the [ROS tutorial](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment) and clone the Git repository into your workspace.

In order to use your trained model, you need to make sure the model is in <ros workspace>/src/hrl_object_detection/src/
and run the following commands:
```
protoc object_detection/protos/*.proto --python\_out=.
```

In <ros workspace>/src/hrl_object_detection/src/models/research:
```
python setup.py build
python setup.py install --user
```

In <ros workspace>/src/hrl_object_detection/src/models/research/slim:
```
python setup.py build
python setup.py install --user
```

In <ros workspace>/src/hrl_object_detection/src/models/research:
  ```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

### Launch ROS nodes
Since there can only be one active model at the same time, the chosen model has to be specified in the code.
To change the model, open the file <ros workspace>/src/hrl_object_detection/src/detect.py in lines 365-367 and
uncomment only the desired model name.
To launch the detection and movement modules and to visualize the robot camera with detections, run each
of the following commands respectively.
```
rosrun hrl_object_detection detect.py
rosrun move_robot control_robot.py
rosrun image_view image_view image:=/image_with_box
```
