import sys
import cv2 as cv
import os
from data_collection.collector          import Data_Collector

img_pth = sys.argv[1]
cam_idx  = int(sys.argv[2])


obj = Data_Collector(img_pth)

img_num = obj.get_img_count(img_pth)

img_name = "img" + str(img_num) + ".png"

img = obj.take_image(img_name=img_name,cam_idx=cam_idx)
cv.imwrite(os.path.join(img_pth,img_name),img)






