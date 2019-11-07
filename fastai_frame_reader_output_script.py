import os
from fastai.vision import *
import numpy as np
import cv2
from PIL import Image, ImageDraw

#from dl1 folder
path = Path('clotting_images')

#TEST_IMAGES_DIR = os.getcwd() + "/clotting_test"

#MODEL_DIR = Path('clotting_images')

classes = ['clot', 'non_clot']
  


#single image
for filename in os.listdir(path/'clotting_test'):
    img = open_image(path/'clotting_test'/filename)
    learn = load_learner(path, file= 'resnet34_02_error_Sept_19_with_transforms_rectangles.pkl')
    
    pred_class,pred_idx,outputs = learn.predict(img)
    print(filename)

    #show image
    show_image = Image.open(path/'clotting_test'/filename)
    show_image.show()
    
    print(pred_class)




#LOAD MODEL for multiple?
#learn = load_learner(clotting_images, test=ImageList.from_folder('PATH'))








#######ITERATE PREDICTIONS
##for fileName in os.listdir(TEST_IMAGES_DIR):
    
#    #run a prediction
#    pred_class,pred_idx,outputs = learn.predict(fileName)
#    pred_class
#    time.sleep(1)
        
    #output the prediction
    
