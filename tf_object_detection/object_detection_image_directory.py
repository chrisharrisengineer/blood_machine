######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from natsort import natsorted, ns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time


#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'crop1.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'images_for_training_and_testing','labelmap.pbtxt')

#PATH_TO_LABELS = os.path.join(CWD_PATH,'images_for_training_and_testing','labelmap.pbtxt')
#print('loaded_path_to_labels')

# Path to REAL analysis cropped_pics
#PATH_TO_IMAGES = '/home/llu-2/Desktop/lluFolder/masterProgram/01_26_19/analysis/cropped_pics/'
#print('loaded_path_to_images')

#TEST IMAGES
#PATH_TO_IMAGES = '/home/llu-2/Desktop/lluFolder/masterProgram/01_26_19/analysis/platelet_test_2/'

#live test images
PATH_TO_IMAGES = 'C:/Users/Predator/Desktop/tensorflow/models/research/object_detection/quick_clotting_test/'

# Path to image
#PATH_TO_IMAGE = os.path.join(CWD_PATH,'analysis/cropped_pics/', IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

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

count = 0

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
#FOR LOOP TO ITERATE THROUGH IMAGE DIRECTORY
for filename in natsorted(os.listdir(PATH_TO_IMAGES)):

	image = cv2.imread(os.path.join(PATH_TO_IMAGES, filename))
	
	image_expanded = np.expand_dims(image, axis=0)
	
	#start timer to get inf time
	start_time = time.time()

	# Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
	    [detection_boxes, detection_scores, detection_classes, num_detections],
	    feed_dict={image_tensor: image_expanded})

	#print time of inference
	count += 1
	print('Iteration %d: %.3f sec'%(count, time.time()-start_time))
	
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

	# All the results have been drawn on image. Now resize and display the image.
	image = cv2.resize(image, (1800,400))
	plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
	plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	plt.show()
	#cv2.imshow('Object detector', image)
	print(filename)	

	#count and display the number of boxes drawn on platelets at each frame
	print(scores)
	print(np.count_nonzero(scores))
	number_of_boxes_drawn.append(np.count_nonzero(scores))	
	print(number_of_boxes_drawn)	
	#fig = px.line(number_of_boxes_drawn)
	#fig.show
	#plt.plot(number_of_boxes_drawn)
	#plt.show()
	#plt.close()

	#fig=go.Figure()
	#fig.add_trace(go.Scatter(y=number_of_boxes_drawn, name='Platelets Per Frame', line=dict(color='firebrick', width=4))) 
	#fig.show()
	#fig.close()

	# Press any key to close the image
	#cv2.waitKey(50)

# Clean up
cv2.destroyAllWindows()
