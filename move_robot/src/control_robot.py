#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from hrl_object_detection.msg import FloatList
from kobuki_msgs.msg import Sound, BumperEvent
from collections import deque
import numpy as np
import time
import math

# Publishers: send velocity and sound to TurtleBot
pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
pub_sound = rospy.Publisher('/mobile_base/commands/sound', Sound, queue_size=10)

x_linear_vel = 0.0      # Linear velocity x
z_angular_vel = 0.0     # Angular velocity z
last_offset = 0.0       # Store last offset detected

area_detected = 0.3     # Threshold area for sending sound

detections = deque(maxlen=7)        # Queue which contains last 7 detections
last_sound_time = time.time()       # Time for spiral sound
last_detection_time = time.time()   # Time for detection sound
spiraling_sound = False             # Spiral sound

maneuvering = False         # If robot is maneuvering after a collision
orientation = 0.0           # Robot orientation
bump_orientation = 0.0      # Orientation when robot collided
desired_orientation = 0.0   # Orientation robot has to achieve
bumper = 0                  # State of bumper  (0 - left, 1 - center, 2 - right)

## Publishes the message for the toy detection
def play_sound(value):
    sound_msg = Sound()
    sound_msg.value = value
    pub_sound.publish(sound_msg)

## Returns a value remaped from one range to another
def remap(old_val, old_min, old_max, new_min, new_max):
    return (new_max - new_min)*(old_val - old_min) / (old_max - old_min) + new_min

## Returns True if there is a majority of detections, False otherwise
def detected_majority(detections):
    pos = 0
    for i in range(0, len(detections)):
        pos = pos + 1 if detections[i] == 1 else pos
    return pos >= len(detections)/2

## Converts a quaternion to euler coordinates
def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]

## Called each time the odometry is updated
## Update TurtleBot orientation
def odom_callback(data):
    global orientation
    orientation_q = data.pose.pose.orientation
    euler = quaternion_to_euler(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
    orientation = euler[0]  # Save yaw

## Called each time the bumper detects a hit
## Calculate desired orientation after the collision
def bumper_callback(data):
    global x_linear_vel, z_angular_vel, bump_orientation, orientation, bumper, desired_orientation, maneuvering, last_offset
    # Data:
    # Bumper: 0 - left, 1- center, 2- right
    # State: 0 - released 1 - pushed
    if data.state == 1 and not maneuvering:
        print("Bumped: ", data)
        bumper = data.bumper

        # Update collision orientation, maneuvering and linear velocity (only rotates)
        bump_orientation = orientation
        maneuvering = True
        x_linear_vel = 0

        # Update angular velocity, last offset (change rotation direction after rotate)
        # and desired_orientation (+/-90º lateral or 180º front collision)
        if data.bumper == 0:
            z_angular_vel = -0.3
            last_offset = 80
            target_rad = (orientation - math.radians(90) + np.pi) % (2 * np.pi) - np.pi
            desired_orientation = target_rad
        elif data.bumper == 1:
            last_offset = -80 if z_angular_vel <= 0 else 80
            z_angular_vel = 0.3
            target_rad = (orientation + math.radians(180) + np.pi) % (2 * np.pi) - np.pi
            desired_orientation = target_rad
        elif data.bumper == 2:
            z_angular_vel = 0.3
            last_offset = -80
            target_rad = (orientation + math.radians(90) + np.pi) % (2 * np.pi) - np.pi
            desired_orientation = target_rad

## Called each time the detection module detects something
## Calculate and updates linear and angular velocities
def detection_callback(data):
    global x_linear_vel, z_angular_vel, detections, last_sound_time, area_detected, last_offset, last_detection_time, spiraling_sound, maneuvering, bump_orientation, orientation

    # Add a detection to the queue
    detections.append(data.detected)

    # If it is not solving a bump collision
    if not maneuvering:
        # If there is a majority in the queue
        if detected_majority(detections):
            last_detection_time = time.time()   # Start timer for exploration
            z_angular_vel = -remap(data.xOffset, -320, 320, -0.3, 0.3)  # Set angular velocity scaled to the offset

            # If the detected box is centered -> Stop rotating
            if data.xOffset >= -50 and data.xOffset <= 50:
                z_angular_vel = 0.0

            last_offset = data.xOffset
            # Go forward scaled to the relative box area
            x_linear_vel = remap(data.area, area_detected, 0, 0.04, 0.1)

            # If the area is greater than the threshold area for detection: stop and beep
            if data.area > area_detected:
                x_linear_vel = 0.0
                if time.time() - last_sound_time > 5.0:
                    last_sound_time = time.time()
                    play_sound(6)
        # If there is no majority in the queue
        else:
            # If object lost, turn towards last seen orientation
            if last_offset > 0:
                z_angular_vel = -0.2
            else:
                z_angular_vel = 0.2
            x_linear_vel = x_linear_vel - 0.02 if x_linear_vel > 0.0 else 0.0

            # If object not found in some time, turn into a spiral to explore
            print("lastdetect time:\t", last_detection_time)
            time_diff = time.time() - last_detection_time
            print("time_diff:\t", time_diff)
            if time_diff > 20.0:
                if not spiraling_sound:
                    play_sound(5)
                    spiraling_sound = True
                x_linear_vel = min(0.1 + 0.0025 * (time_diff-20.0), 0.25)
                # Reduce velocity to make spiral bigger
                if last_offset > 0:
                    z_angular_vel = -0.125
                else:
                    z_angular_vel = 0.125
            else:
                spiraling_sound = False

        print("detected:\t", bool(data.detected))
        print("offset:\t\t", data.xOffset)
        print("last offset:\t", last_offset)
        print("rotvel:\t\t", z_angular_vel)
        print("area:\t\t" , data.area)
        print("linvel:\t\t", x_linear_vel)
        print("detections:\t", detections)
        print("majority:\t", detected_majority(detections))

        print("--------------------------------------------------")
        #rospy.loginfo(rospy.get_caller_id() + 'I heard %f and %f', data.xOffset, data.area)
    else:
        print("Maneuvering")
        print("bump_orientation:\t", bump_orientation)
        print("Orientation:\t\t", orientation)
        print("Desired Orientation: \t", desired_orientation)
        print("diff:\t\t",abs(desired_orientation - orientation))
        # Bumper: 0 - left, 1- center, 2- right
        if abs(desired_orientation - orientation) < 0.2 or detected_majority(detections):
            last_detection_time = time.time() - 21.0
            maneuvering = False

## Publishes updated velocities to the robot controller
def control_robot():
    rospy.loginfo("To stop TurtleBot CTRL + C")

    # initiliaze node
    rospy.init_node('control_robot', anonymous=True)

    rate = rospy.Rate(10)   # How often should move (10 HZ)
    move_cmd = Twist()      # Datatype for velocity

    rospy.Subscriber('control_robot/camera_detection', FloatList, detection_callback)
    rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, bumper_callback)
    rospy.Subscriber('/odom', Odometry, odom_callback)

    while not rospy.is_shutdown():
        # Send velocity to TurtleBot
        move_cmd.linear.x = x_linear_vel
        move_cmd.angular.z = z_angular_vel
        pub.publish(move_cmd)

        # wait for 0.1 seconds, 10 HZ
        rate.sleep()

if __name__ == '__main__':
    try:
        control_robot()
    except:
        rospy.loginfo("control_robot node terminated.")
