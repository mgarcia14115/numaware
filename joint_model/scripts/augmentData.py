import os

import sys
import albumentations as A
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import random
from data_collection.collector          import Data_Collector
import numpy as np
import csv

dir_path = sys.argv[1]
data_path = sys.argv[2]

obj = Data_Collector()

#https://albumentations.ai/docs/getting_started/transforms_and_targets/
transformations = [
    A.AdditiveNoise(p=1),
    A.Blur(blur_limit=12,p=1),
    A.CLAHE(p=1),
    A.ChromaticAberration(p=1),
    A.Defocus(p=1),
    A.Downscale(p=1),
    A.Emboss(p=1),
    A.GaussNoise(p=1),
    A.GaussianBlur(p=1),
    A.Illumination(p=1),
    A.MedianBlur(p=1),
    A.MotionBlur(p=1),
    A.MultiplicativeNoise(p=1),
    A.SaltAndPepper(p=1),
    A.PlasmaShadow(p=1),
    A.Posterize(p=1),
    A.Superpixels(p=1),
    A.ShotNoise(p=1)
]
transformations_list = [
    "additivenoise",
    "blur",
    "clahe",
    "chromaticaberration",
    "defocus",
    "downscale",
    "emboss",
    "gaussnoise",
    "gaussianblur",
    "illumination",
    "medianblur",
    "motionblur",
    "multiplicativenoise",
    "saltandpepper",
    "plas_shadow",
    "posterize",
    "superpixels",
    "shotnoise"
]





def transform_img(transformations,path_dir,image,img_num):
    idx = 0
    for trans in transformations:
        
        transform = A.Compose(trans)
     
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        transformed_image = cv.cvtColor(transformed_image, cv.COLOR_BGR2RGB) 
        aug_folder = path_dir+ "/"+ transformations_list[idx]
        
        os.makedirs(aug_folder, exist_ok=True)
        
        cv.imwrite(aug_folder + "/img"+ img_num + "_"+ transformations_list[idx]+  ".png",transformed_image)
        
        idx+=1
   

def findImg(all_img,img_key):
    
    for img in all_img:

        img_name = img[0:img.index("_")]
        if img_key == img_name:
            return img
    return -1
    



for file in os.listdir(dir_path):
    
    if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):     
        image = cv.imread(os.path.join(dir_path,file))
        img_num = file[file.index("g") + 1 : file.index(".")]
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        transform_img(transformations,dir_path,image,img_num)



# with open(data_path,mode='a+', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     writer = csv.writer(csvfile) # Create writer to append to file
#     next(reader, None)           #Skip the header
#     csvfile.seek(0, os.SEEK_END)
#     for row in reader:
#         img_name = row[0]
#         original_img = img_name[0:img_name.index(".")] #imgNum
        
#         for dir in os.listdir(dir_path): # Iterate throught content of directories
            
#             pth = os.path.join(dir_path,dir)
#             if os.path.isdir(pth): # Subset only folders                
#                 new_img_name = findImg(os.listdir(pth),original_img) # Find matching image
#                 if new_img_name != -1:
#                     row[0] = os.path.join(dir,new_img_name)
#                     writer.writerow(row) # append row along with its path

# Open the CSV file for reading first
with open(data_path, mode='r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)  # Read all rows into memory

# Now, open the file again in append mode for writing
with open(data_path, mode='a', newline='') as csvfile:
    writer = csv.writer(csvfile)  # Create the writer

    # Skip the header if it exists
    header = rows[0]  # Store header
    writer.writerow(header)  # Write header (only once)
    
    for row in rows[1:]:  # Skip header row and process others
        img_name = row[0]
        original_img = img_name[0:img_name.index(".")]  # imgNum

        # Iterate through directories to find matching images
        for dir in os.listdir(dir_path):
            pth = os.path.join(dir_path, dir)
            if os.path.isdir(pth):  # Only directories
                new_img_name = findImg(os.listdir(pth), original_img)  # Find matching image
                if new_img_name != -1:
                    row[0] = os.path.join(dir, new_img_name)
                    writer.writerow(row)  # Append the row along with its new path

                
                
       
            
