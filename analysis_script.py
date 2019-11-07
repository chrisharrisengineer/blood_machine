#use python 3.6 - run from fast.ai environment
#!home/llu-2/anaconda3/envs/fastai/bin/python

import os
from fastai.vision import *
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2

from natsort import natsorted, ns
import time

#path to all
path = Path('analysis')

#path for non live testing
#picture_path = Path('clotting_test_2')

#path for live test
picture_path= Path('cropped_pics')

#path_to_IMG = Path(/home/llu-2/Desktop/lluFolder/masterProgram/01_26_19/cropped)


classes = ['clot', 'non_clot']

full_clot_probability_array = []
count = 0

full_path = Path("analysis/cropped_pics")

				
#for filename in sorted(os.listdir(path/picture_path), key=lamba)
for filename in natsorted(os.listdir(path/picture_path)):
	
	img = open_image(path/picture_path/filename)

	learn = load_learner(path, file='resnet34_02_error_Sept_19_with_transforms_rectangles.pkl')
  
	pred_class,pred_idx,outputs = learn.predict(img)
	print(filename)

	#pred_class2,pred_idx2,outputs2 = learn.predict(imgM2)	
	
	this_image = os.path.join('/home/bha/Desktop/lluFolder/masterProgram/01_26_19/analysis/cropped_pics', filename)
	image_read = cv2.imread(this_image)
	resized_image = cv2.resize(image_read, (1600, 400))	
	cv2.imshow('tube', resized_image)	
	cv2.waitKey(1)

	#store probabilities, when probability of 4/5 frames is over 80% a clot, call it at that 		time; first number is prob clot, second is prob non_clot		
	#take last five (output values), add all up, / by 5, get at least 80%, then output that 	time	
	
	#convert torch.Tensor to nparray
	print(type(outputs))
	tensor_to_np = outputs.numpy()
	print(tensor_to_np)

	#grab clot probability from each output
	grab_clot_probability = tensor_to_np[0]
	print(grab_clot_probability)
	
	#add each new clot probability to end of array
	
	full_clot_probability_array.append(grab_clot_probability)
	print(full_clot_probability_array)	
	    
	plt.plot(full_clot_probability_array)	
	plt.draw()
	plt.pause(.001)	

	#take last 5 clot probabilities in this array, add them up, divide by 5, if equal over .8, call a clot
	last_five_array = full_clot_probability_array[-5:]
	sum_last_five = sum(last_five_array)
	last_five_avg = sum_last_five/5 
	last_five_avg = float(last_five_avg)
	print("avg is" + str(last_five_avg))

	

	if last_five_avg > .8:
	    print("the blood has clotted")		
	else:
	    print("the blood has not clotted")	
	

	#return pred_class1, pred_class2
	
	#print(pred_class)
	#print(pred_idx)
	print(outputs)
	#print(pred_class2)
	#print(clot_probability)

	count = count + 1
