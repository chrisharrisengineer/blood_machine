#!/home/bha/anaconda3/envs/tf_gpu/bin/python bash

######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Chris Harris
# Date: 5/15/19
# Description: 
# Object Detection of platelets using tensorflow

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from natsort import natsorted, ns
import matplotlib
matplotlib.get_backend() 
import matplotlib.pyplot as plt
matplotlib.get_backend() 
matplotlib.use('TKAgg')
matplotlib.get_backend() 
#import plotly.graph_objects as go
import time
from scipy.interpolate import *
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.filters import hp_filter



# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph_platelet_dec_6_2019'
IMAGE_NAME = 'crop1.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join('/home/bha/Desktop/lluFolder/tensorflow1/models/research/object_detection',MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join('/home/bha/Desktop/lluFolder/masterProgram/01_26_19','images_for_training_and_testing','labelmap.pbtxt')

#PATH_TO_LABELS = os.path.join(CWD_PATH,'images_for_training_and_testing','labelmap.pbtxt')
#print('loaded_path_to_labels')

# Path to REAL analysis cropped_pics
#PATH_TO_IMAGES = '/home/llu-2/Desktop/lluFolder/masterProgram/01_26_19/analysis/cropped_pics/'
#print('loaded_path_to_images')

#TEST IMAGES
#PATH_TO_IMAGES = '/home/llu-2/Desktop/lluFolder/masterProgram/01_26_19/analysis/platelet_test_2/'

#live test images
PATH_TO_IMAGES = '/home/bha/Desktop/lluFolder/masterProgram/01_26_19/analysis/cropped_pics/'

# Path to image
#PATH_TO_IMAGE = os.path.join(CWD_PATH,'analysis/cropped_pics/', IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1


# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#array, custom num of detections using scores
number_of_boxes_drawn = []

frame_number = 0
endpoint = []
frame_endpoint_time = []
endpoint_final_frame_number = []

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
#FOR LOOP TO ITERATE THROUGH IMAGE DIRECTORY
for filename in natsorted(os.listdir(PATH_TO_IMAGES)):
	print(filename)
	image = cv2.imread(os.path.join(PATH_TO_IMAGES, filename))
	
	image_expanded = np.expand_dims(image, axis=0)
	
	#start timer to get inf time
	start_time = time.time()
	print('Detection start time:  Iteration %d: %.3f sec'%(frame_number, time.time()-start_time))
	# Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
	    [detection_boxes, detection_scores, detection_classes, num_detections],
	    feed_dict={image_tensor: image_expanded})

	#print time of inference
	frame_number += 1
	print('Detection End Time:  Iteration %d: %.3f sec'%(frame_number, time.time()-start_time))
	
	# Draw the results of the detection (aka 'visualize the results')

	vis_util.visualize_boxes_and_labels_on_image_array(
    	    image,
    	    np.squeeze(boxes),
    	    np.squeeze(classes).astype(np.int32),
    	    np.squeeze(scores),
       	    category_index,
            use_normalized_coordinates=True,
    	    line_thickness=8,
       	    min_score_thresh=0.60)

	print('Draw time: Iteration %d: %.3f sec'%(frame_number, time.time()-start_time))

	# All the results have been drawn on image. Now resize and display the image.
	image = cv2.resize(image, (1800,400))
	cv2.imshow('Object detector', image)
	print(filename)	
	print('Display Image: Iteration %d: %.3f sec'%(frame_number, time.time()-start_time))
	

	#count and display the number of boxes drawn on platelets at each frame
	print(scores)
	print(np.count_nonzero(scores))
	number_of_boxes_drawn.append(np.count_nonzero(scores))	
	print(number_of_boxes_drawn)	
	
	print('Count Boxes: Iteration %d: %.3f sec'%(frame_number, time.time()-start_time))

	plt.plot(number_of_boxes_drawn)	
	plt.draw()
	plt.pause(.001)

	#smooth graph/remove noise with hp filter; store the smoothed values in "trend"
	if frame_number > 50:
		#cycle, trend = hp_filter.hpfilter(number_of_boxes_drawn, lamb=50)
		#cycle, trend = hp_filter.hpfilter(number_of_boxes_drawn, lamb=5800)
		cycle, trend = hp_filter.hpfilter(number_of_boxes_drawn, lamb=50000)
		plt.plot(trend)
		plt.draw()
		plt.pause(.001)
		plt.clf()

	#how to find where there is a + slope change for a long time(add 50 frames gradient)
	#find the max gradient	
	if frame_number > 50: 
		if not endpoint and ((trend[frame_number-1]-trend[frame_number-50]) > .1):
            #print("in endpoint loop")
            #print(trend[frame_number-1])
            #append slope to endpoint list
			endpoint.append(trend[frame_number-1])
			endpoint_final_frame_number = frame_number - 50 
			count = 0                   
			for frame in natsorted(os.listdir(PATH_TO_IMAGES)):
				count = count + 1                 
				if count == endpoint_final_frame_number:
					file = frame 
					position = file.index('.')
                    #gets filename position after first (.) and before last(.)                     
					clotting_endpoint_time = str(file[position + 1 :position +3])
                    #print("clotting endpoint time is")
                    #print(clotting_endpoint_time)
					stdout = sys.stdout.write(str(clotting_endpoint_time))
					
	#how to find where there is a slope change for a long time
	#find the max gradient



	# Press any key to close the image
	cv2.waitKey(50)
	print('Total loop time: Iteration %d: %.3f sec'%(frame_number, time.time()-start_time))
	

# Clean up
cv2.destroyAllWindows()
