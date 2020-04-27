#!/bin/bash

source /opt/ros/melodic/setup.bash 

source devel/setup.bash 

export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/catkin_ws_new/src

export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/catkin_ws_new/

cd src/hrl_object_detection/src/models/research

protoc object_detection/protos/*.proto --python_out=.

python3 setup.py build

python3 setup.py install --user

cd slim/

python3 setup.py build

python3 setup.py install --user

cd ..

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

cd ~/catkin_ws_new

