#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import Image


bridge = CvBridge()

## Callback function called everytime the listener gets a message
def callback(data):
    image = bridge.imgmsg_to_cv2(data, "bgr8")
    #rospy.loginfo(data)
    print(image)
    cv2.imshow("pepino", image)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()

## This declares the listener part of the node
def listener():
    print("pillamo fotaca")
    
    rospy.init_node('tf_node', anonymous=True)

    #rospy.Subscriber('control_robot/camera_detection', String, callback)
    rospy.Subscriber('camera/rgb/image_rect_color', Image, callback)
    rospy.spin()

if __name__ == '__main__':
    rospy.loginfo('holi')
    listener()

    
