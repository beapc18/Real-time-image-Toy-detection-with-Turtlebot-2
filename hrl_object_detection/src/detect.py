#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from hrl_object_detection.msg import FloatList
from sensor_msgs.msg import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# Bridge to convert images
bridge = CvBridge()

### VARIABLES
path_to_labels_platypus = "/home/user/sanchezs/catkin_ws/src/hrl_object_detection/src/platypus/platypus-detection.pbtxt"
path_to_labels_unicorn = "/home/user/sanchezs/catkin_ws/src/hrl_object_detection/src/unicorn/unicorn_object-detection.pbtxt"
path_to_labels_teddy_monster = "/home/user/sanchezs/catkin_ws/src/hrl_object_detection/src/teddy_monster/teddy_monster_object-detection.pbtxt"

model_name_platypus = "/home/user/sanchezs/catkin_ws/src/hrl_object_detection/src/platypus/platypus_inference_graph"
model_name_unicorn = "/home/user/sanchezs/catkin_ws/src/hrl_object_detection/src/unicorn/unicorn_graph"
model_name_teddy_monster = "/home/user/sanchezs/catkin_ws/src/hrl_object_detection/src/teddy_monster/teddy_monster_inference_graph"

class_to_detect_platypus = "platypus"
class_to_detect_unicorn = "unicorn"
class_to_detect_teddy_monster = "teddy monster"

detected_top_camera = 0


#dominant_colors_unicorn = np.float32(np.array([[20.04054069519043, 16.291893005371094, 22.2891902923584],
                                               #[77.27994537353516, 72.75, 86.98204040527344],
                                               #[110.84375, 104.1611328125, 113.310546875],
                                               #[167.5294952392578, 153.17245483398438, 180.65354919433594],
                                               #[213.70997619628906, 211.31106567382812, 214.26806640625]]))
#dominant_colors_platypus = np.float32(np.array([[7.902225017547607, 10.378686904907227, 18.599586486816406],
                                                #[39.54244613647461, 47.89728546142578, 65.26145935058594],
                                                #[64.30582427978516, 99.69781494140625, 105.12933349609375],
                                                #[104.90493774414062, 104.2941665649414, 135.60922241210938],
                                                #[129.13047790527344, 164.89979553222656, 190.93841552734375]]))
#dominant_colors_monster = np.float32(np.array([[20.952489852905273, 20.191192626953125, 22.156431198120117],
                                               #[65.02598571777344, 83.71105194091797, 86.01863861083984],
                                               #[85.77362823486328, 108.91241455078125, 135.65928649902344],
                                               #[154.9387664794922, 157.45663452148438, 157.8596954345703],
                                               #[212.35440063476562, 214.16761779785156, 214.51052856445312]]))
dominant_colors_unicorn = np.float32(np.array([[14.512475967407227, 10.637235641479492, 12.238003730773926],
                                               [45.42377853393555, 35.51829147338867, 39.001522064208984], [73.22736358642578, 56.2325553894043, 64.88287353515625],
                                               [93.06961822509766, 62.49015808105469, 91.50210571289062],
                                               [101.87042999267578, 79.54219055175781, 140.76304626464844],
                                               [133.23292541503906, 84.64583587646484, 146.19268798828125], 
                                               [146.9541778564453, 121.57429504394531, 183.50267028808594],
                                               [175.8449249267578, 130.68478393554688, 196.73333740234375],
                                               [189.53985595703125, 174.63101196289062, 214.19927978515625], 
                                               [217.00360107421875, 216.41111755371094, 217.29551696777344]]))
dominant_colors_platypus = np.float32(np.array([[5.150073051452637, 5.405563831329346, 8.959736824035645],
                                                [19.21808433532715, 22.307445526123047, 29.933433532714844], 
                                                [37.59239959716797, 38.90922546386719, 33.54680633544922],
                                                [55.65052795410156, 53.83592224121094, 56.52360534667969],
                                                [65.02174377441406, 68.0908432006836, 76.18840789794922],
                                                [87.24678039550781, 87.39642333984375, 78.73574829101562],
                                                [100.889404296875, 89.90760803222656, 118.34420776367188],
                                                [107.12446594238281, 131.53529357910156, 157.36705017089844], 
                                                [132.6621551513672, 161.353271484375, 183.85906982421875],
                                                [210.23611450195312, 211.0833282470703, 211.4166717529297]]))
dominant_colors_monster = np.float32(np.array([[8.104718208312988, 5.26467227935791, 5.033371925354004],
                                               [32.14719772338867, 23.15839958190918, 19.88159942626953], [54.60653305053711, 42.97511672973633, 38.53810501098633],
                                               [74.2969741821289, 60.45454788208008, 51.90129852294922], 
                                               [88.74835205078125, 71.97657775878906, 63.67147445678711],
                                               [107.48922729492188, 87.00760650634766, 76.51457214355469],
                                               [138.4444580078125, 121.26786041259766, 113.48809814453125],
                                               [151.0196075439453, 150.47964477539062, 150.7978973388672], 
                                               [188.70578002929688, 186.58016967773438, 182.24627685546875],
                                               [216.57745361328125, 216.20333862304688, 215.73629760742188]]))
dominant_colors_active = 0 

area_detected_platypus = 0.1
area_detected_unicorn = 0.08
area_detected_monster = 0.1
area_detected = 0 # area_detected_monster

detection_offset = 0
detection_relative_area = 0
validated_relative_area = 0

detected = 0
update = False
detection_model = None

class_to_detect = ""
path_to_labels = ""
category_index = None
detection_threshold = 0.6

validating = 0
detection_validated = 0


pub_image = rospy.Publisher('/image_with_box', Image, queue_size=10)


### FUNCTIONS
## Returns the tensorflow model loaded
def load_model(model_name):
    model_dir = model_name
    model_dir = Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir), None)
    model = model.signatures["serving_default"]

    return model


## Return True if the color pattern matches the toy wanted
def is_my_toy(dominants):
    global dominant_colors_platypus, dominant_colors_unicorn, dominant_colors_monster
    
    dominants = np.sort(dominants, axis=0)
    
    diff_platypus = np.average(np.absolute(dominants - dominant_colors_platypus))
    diff_unicorn = np.average(np.absolute(dominants - dominant_colors_unicorn))
    diff_monster = np.average(np.absolute(dominants - dominant_colors_monster))
    print("Avg error platypus:\t", diff_platypus)
    print("Avg error unicorn:\t", diff_unicorn)
    print("Avg error monster:\t", diff_monster)
    if model == "platypus":
        return diff_platypus < diff_monster and diff_platypus < diff_unicorn
    elif model == "unicorn":
        return diff_unicorn < diff_monster and diff_unicorn < diff_platypus
    elif model == "teddy_monster":
        return diff_monster < diff_platypus and diff_monster < diff_unicorn

## Returns the palette of dominant colors from an image
def get_dominant_colors(image, box):
    top, left, bottom, right = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    box_img = image[top:bottom,left:right]
    
    pixels = np.float32(box_img.reshape(-1, 3))

    n_colors = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    
    return palette
    
## Runs one inference pass for the image
def run_inference_for_single_image(model, image):
    # Convert it to numpy array
    image = np.asarray(image)
    
    # The input needs to be a tensor
    input_tensor = tf.convert_to_tensor(image)
    
    # The model expects a batch of images, so add an axis
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    output_dict = model(input_tensor)
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(output_dict['detection_masks'], output_dict['detection_boxes'], image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

## Returns detected box width, height, horizontal center and area
def calc_box_values(box, shape):
    box[0] *= shape[0]
    box[1] *= shape[1]
    box[2] *= shape[0]
    box[3] *= shape[1]
    
    detected_box_width = box[3] - box[1]
    detected_box_height = box[2] - box[0]
    detected_box_centerX = box[1] + (detected_box_width/2.0)
    detected_box_area = detected_box_width * detected_box_height
    return detected_box_width, detected_box_height, detected_box_centerX, detected_box_area

## Adjusts gamma and brightness for the given image
def adjust(image, gamma=1.0, brightness=1):
    hsvImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # decreasing the V channel by a factor from the original
    hsvImg[...,2] = hsvImg[...,2]*brightness
    
    new_image = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(new_image, table)

    
## Callback function called everytime the listener gets a detection message from the lower camera
def detect_callback(data):
    print("Validating:\t", validating)
    global detection_offset, detection_relative_area, detected

    image = adjust(bridge.imgmsg_to_cv2(data, "bgr8"), gamma=0.7, brightness=0.9)
    
    image_centerX = image.shape[1] / 2.0
    image_area = image.shape[0] * image.shape[1]
    image_to_publish = image.copy()

    output_dict = run_inference_for_single_image(detection_model, image)
    
    ## Box detection_boxes: [top, left, bottom, right]
    ## Detection_scores: value between 0 and 1
    ## Detection_classes: The name of the class we are detecting
    detected_boxes = []
    detected_scores = []

    # First, filter the predictions that are not the class we are interested in
    for i, entry in enumerate(output_dict['detection_classes']):
        # print(category_index[entry])
        if category_index[entry]["name"] == class_to_detect:
            detected_boxes.append(output_dict['detection_boxes'][i])
            detected_scores.append(output_dict['detection_scores'][i])

    if detected_scores:
        # Second, check which one of those detections has the higher score
        max_index = detected_scores.index(max(detected_scores))
        print("Score:\t\t", detected_scores[max_index], "\r")
        print("--------------------------------------------\t")
        # print(detected_scores[max_index])
        # Third, if that score is higher than a threshold, we compute the values to send to the controller
        if detected_scores[max_index] >= detection_threshold:
            detected_box = detected_boxes[max_index]
    
            
            detected_box_width, detected_box_height, detected_box_centerX, detected_box_area = calc_box_values(detected_box, image.shape)

            # Update values that we need to send
            detection_offset = detected_box_centerX - image_centerX # If positive, the box is on the right side of the image. If negative, the box is on the left side of the image
            detection_relative_area = detected_box_area / image_area # Value between 0 and 1 to check if we are close or far away from the object. The closer we are, the bigger the box will be
            print("area:\t\t", detection_relative_area)
            dominant_colors = get_dominant_colors(image, detected_box)
            #print("Dominant colors: \n: ", dominant_colors)
            #print("Dominant colors: \n: ", np.sort(dominant_colors, axis=0).tolist())
            ismytoy = is_my_toy(dominant_colors)
            
            if (ismytoy):
                detected = 1
            else:
                detected = 0
                
            image_to_publish = cv2.rectangle(image_to_publish, (detected_box[1], detected_box[0]), (detected_box[3], detected_box[2]), (0, 255, 0) if detected else (0, 0, 255), 2)
        else:
            detected = 0
    else:
        detected = 0
    
    # publicar la imagen
    if not validating:
        pub_image.publish(bridge.cv2_to_imgmsg(image_to_publish, "bgr8"))

## Callback function called everytime the listener gets a detection message from the upper camera
def validate_detection_callback(data):
    global detection_offset, detection_relative_area, validated_relative_area, detected, validating, detection_validated, detected_top_camera
    
    
    if (detected == 1 and detection_relative_area > area_detected) or validated_relative_area > 0.3:
        validating = 1
        image = bridge.imgmsg_to_cv2(data, "bgr8")
        image_area = image.shape[0] * image.shape[1]
        image_to_publish = image.copy()
        image_centerX = image.shape[1] / 2.0

        output_dict = run_inference_for_single_image(detection_model, image)
        
        ## Box detection_boxes: [top, left, bottom, right]
        ## Detection_scores: value between 0 and 1
        ## Detection_classes: The name of the class we are detecting
        detected_boxes = []
        detected_scores = []

        # First, filter the predictions that are not the class we are interested in
        for i, entry in enumerate(output_dict['detection_classes']):
            # print(category_index[entry])
            if category_index[entry]["name"] == class_to_detect:
                detected_boxes.append(output_dict['detection_boxes'][i])
                detected_scores.append(output_dict['detection_scores'][i])
        
        if detected_scores:
            # Second, check which one of those detections has the higher score
            max_index = detected_scores.index(max(detected_scores))
            print("Confirmation Score:\t", detected_scores[max_index], "\r")
            print(".......\r")
            # Third, if that score is higher than a threshold, we compute the values to send to the controller
            if detected_scores[max_index] >= detection_threshold:
                detected_top_camera = 1
                detected_box = detected_boxes[max_index]
                
                detected_box_width, detected_box_height, detected_box_centerX, detected_box_area = calc_box_values(detected_box, image.shape)
                detection_offset = detected_box_centerX - image_centerX # If positive, the box is on the right side of the image. If negative, the box is on the left side of the image

                validated_relative_area = detected_box_area / image_area # Value between 0 and 1 to check if we are close or far away from the object. The closer we are, the bigger the box will be
                print("Relative area:\t", validated_relative_area)
                if validated_relative_area > 0.3:
                    detection_validated = 1
                else:
                    detection_validated = 0
                image_to_publish = cv2.rectangle(image_to_publish, (detected_box[1], detected_box[0]), (detected_box[3], detected_box[2]), (0, 255, 0), 2)
                
            else:
                detected_top_camera = 0
        else:
            detected_top_camera = 0
        pub_image.publish(bridge.cv2_to_imgmsg(image_to_publish, "bgr8"))
    else:
        validating = 0


## This declares the listener part of the node
def listener():
    rospy.Subscriber('camera/rgb/image_raw', Image, detect_callback)
    rospy.Subscriber('/camera_sr300/color/image_raw', Image, validate_detection_callback)


## This declares the publisher part of the node
def publisher():
    global detection_offset, detection_relative_area, validated_relative_area

    pub = rospy.Publisher('/control_robot/camera_detection', FloatList, queue_size=10)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        msg = FloatList()
        msg.area = validated_relative_area
        msg.xOffset = detection_offset
        msg.detected = detected if not validating else detected_top_camera
        pub.publish(msg)
        rate.sleep()


#model = "platypus"
model = "teddy_monster"
#model= "unicorn"


if __name__ == '__main__':
    #global detection_model, path_to_labels, model_name, class_to_detect, category_index

    if model == "platypus":
        path_to_labels = path_to_labels_platypus
        model_name = model_name_platypus
        class_to_detect = class_to_detect_platypus
        area_detected = area_detected_platypus
    elif model == "unicorn":
        path_to_labels = path_to_labels_unicorn
        model_name = model_name_unicorn
        class_to_detect = class_to_detect_unicorn
        area_detected = area_detected_unicorn
    elif model == "teddy_monster":
        path_to_labels = path_to_labels_teddy_monster
        model_name = model_name_teddy_monster
        class_to_detect = class_to_detect_teddy_monster
        area_detected = area_detected_monster
    else:
        print("ERROR: Specified model doesn't exist!")

    # List of the strings that is used to add correct label for each box.
    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

    detection_model = load_model(model_name)

    rospy.init_node('tf_node', anonymous=True)

    listener()

    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
