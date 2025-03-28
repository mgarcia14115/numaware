import os

import sys
import albumentations as A
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import random
from collector import Data_Collector
import numpy as np

dir_path = sys.argv[1]

obj = Data_Collector()

#https://albumentations.ai/docs/getting_started/transforms_and_targets/
transformations = [
    A.CLAHE(),
    A.Blur(blur_limit=12),
    A.OpticalDistortion(),
    A.GridDistortion(),
    A.HueSaturationValue(),
    A.ElasticTransform(alpha=1, sigma=50, p=0.5),
    A.ToGray(),
    A.ZoomBlur(),
    A.RGBShift(),
    A.ToSepia(),
    A.SaltAndPepper(),
    A.ShotNoise(),
    A.AutoContrast()
]




def transform_img(transformations,path_dir,image,num):
    
    for trans in transformations:
        transform = A.Compose(trans)
        random.seed(42)
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        cv.imwrite(path_dir + "/aug/img" + str(num) + ".png",transformed_image)
        num+=1

    return num

num = obj.get_img_count(dir_path) + 1
print(dir_path)
for file in os.listdir(dir_path):
    
    if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):     
        image = cv.imread(os.path.join(dir_path,file))
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        num = transform_img(transformations,dir_path,image,num)
