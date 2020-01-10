#use python 3.6 - run from fast.ai environment
#!/home/bha/fastai36/bin/activate
import os
import sys
from fastai.vision import *
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2
from statsmodels.tsa.filters import hp_filter
from natsort import natsorted, ns
import time
from __main__ import *

#path to all
path = Path('analysis')

#path for non live testing
#picture_path = Path('clotting_test_2')

#path for live test
picture_path= Path('cropped_pics')

#path_to_IMG = Path(/home/llu-2/Desktop/lluFolder/masterProgram/01_26_19/cropped)


classes = ['clot', 'non_clot']

full_clot_probability_array = []
frame_number = 1

full_path = Path("analysis/cropped_pics")

endpoint = []
frame_endpoint_time = []
endpoint_final_frame_number = []
#clotting_endpoint_time = []

#numpy rolling averages function
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

clotting_endpoint_time = 0
num_analysis_done = 0
 

while True:
    if num_analysis_done == len(os.listdir(path/picture_path)):
        time.sleep(1)
        if num_analysis_done == len(os.listdir(path/picture_path)):
            time.sleep(1)
            if num_analysis_done == len(os.listdir(path/picture_path)):
                break

    files = natsorted(os.listdir(path/picture_path))
    filename = files[num_analysis_done]   

    img = open_image(path/picture_path/filename)
    learn = load_learner(path, file='resnet34_02_error_Sept_19_with_transforms_rectangles.pkl')
  
    pred_class,pred_idx,outputs = learn.predict(img)
    #print(filename)

    #pred_class2,pred_idx2,outputs2 = learn.predict(imgM2)

    #this_image = os.path.join('/home/bha/Desktop/lluFolder/masterProgram/01_26_19/analysis/cropped_pics', filename)
    this_image = os.path.join('/home/bha/Desktop/lluFolder/masterProgram/01_26_19/analysis/cropped_pics', filename)
    image_read = cv2.imread(this_image)
   

    #resized_image = cv2.resize(image_read, (1800, 400))
    resized_image = cv2.resize(image_read, (1200, 250))
    cv2.imshow('tube', resized_image)
    cv2.waitKey(1)
    
    
    #convert torch.Tensor to nparray
    #print(type(outputs))
    tensor_to_np = outputs.numpy()
    #print(tensor_to_np)
    
    #grab clot probability from each output
    grab_clot_probability = tensor_to_np[0]
    #print(grab_clot_probability)
    
    #add each new clot probability to end of array
      
    full_clot_probability_array.append(grab_clot_probability)
    #print("probability array is")
    #print(full_clot_probability_array)    
        
    plt.plot(full_clot_probability_array)
    plt.draw()
    plt.pause(.001)    
    
    #smooth graph/remove noise with hp filter; store the smoothed values in "trend", and plot on same graph
    if frame_number > 1:
        cycle, trend = hp_filter.hpfilter(full_clot_probability_array, lamb=100000)
        #print("trend is")
        #print(trend)
        plt.plot(trend)
        plt.draw()
        plt.pause(.001)
        plt.clf()
    
    #numpy rolling average
    #if frame_number > 1:
    #    clot_moving_average = movingaverage(full_clot_probability_array, 20)
    #    plt.plot(clot_moving_average)
    #    plt.draw()
    
    #if frame_number > 1:
    #    clot_moving_average = movingaverage(full_clot_probability_array, 100)
    #    plt.plot(clot_moving_average)
    #    plt.draw()
                    
    #check if endpoint array is empty and if slope is greater than ___ from this frame to 50 frames back
    if frame_number > 50:        
        if not endpoint and ((trend[frame_number-1]-trend[frame_number-50]) > .3):
            #print("in endpoint loop")
            #print(trend[frame_number-1])
            #append slope to endpoint list
            endpoint.append(trend[frame_number-1])
            endpoint_final_frame_number = frame_number - 50            
            #extract time from that frame number in cropped_pics folder
            #go to that folder(cropped_pics), grab endpoint frame, endpoint is a number, go to ____ item in folder
            count = 0                   
            for frame in natsorted(os.listdir(path/picture_path)):
                count = count + 1                 
                if count == endpoint_final_frame_number:
                    file = frame 
                    position = file.index('.')
                    #gets filename position after first (.) and before last(.)                     
                    clotting_endpoint_time = str(file[position + 1 :position +3])
                    #print("clotting endpoint time is")
                    #print(clotting_endpoint_time)
                    stdout = sys.stdout.write(str(clotting_endpoint_time))
                         
                
            #print("endpoint is")
            #print(endpoint)
            #print("endpoint frame time is")                    
    
    
            #plot endpoint line on graph and get and save endpoint time (update this loop to plot and find endpoint only once and save, not every time)
    if endpoint:
        #print("endpoint frame is")  
        #print(endpoint_final_frame_number)       
        plt.axvline(x=endpoint_final_frame_number, color='r', linestyle='--')
        
    #return pred_class1, pred_class2
    #print(pred_class)
    #print(pred_idx)
    #print(outputs)
    #print(pred_class2)
    #print(clot_probability)

    num_analysis_done = num_analysis_done +1
    frame_number = frame_number + 1
