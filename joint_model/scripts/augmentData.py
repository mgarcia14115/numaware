import os

import sys
import albumentations as A
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import random
from data_collection.collector          import Data_Collector
import numpy as np

dir_path = sys.argv[1]

obj = Data_Collector()

#https://albumentations.ai/docs/getting_started/transforms_and_targets/
transformations = [
    A.AdditiveNoise(),
    A.AdvancedBlur(),
    A.Blur(blur_limit=12),
    A.CLAHE(),
    A.ChromaticAberration(),
    A.Defocus(),
    A.Downscale(),
    A.Emboss(),
    A.GaussNoise(),
    A.GaussianBlur(),
    A.Illumination(),
    A.MedianBlur(),
    A.MotionBlur(),
    A.MultiplicativeNoise(),
    A.SaltAndPepper(),
    A.PlasmaShadow(),
    A.Posterize(),
    A.Superpixels(),
    A.ShotNoise()
]
transformations_list = [
    "additivenoise",
    "advancedblur",
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





def transform_img(transformations,path_dir,image):
    idx = 0
    for trans in transformations:
        
        transform = A.Compose(trans)
        random.seed(42)
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        transformed_image = cv.cvtColor(transformed_image, cv.COLOR_BGR2RGB) 
        aug_folder = path_dir+ "/"+ transformations_list[idx]
        os.makedirs(aug_folder, exist_ok=True)
        cv.imwrite(aug_folder + "/img"+ "_" + transformations_list[idx]+  ".png",transformed_image)
        idx+=1
   

    


print(dir_path)
for file in os.listdir(dir_path):
    
    if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):     
        image = cv.imread(os.path.join(dir_path,file))
        img_num = file[file.index("g") + 1 : file.index(".")]
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        print(img_num)
        #transform_img(transformations,dir_path,image)
