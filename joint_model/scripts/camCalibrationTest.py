
import cv2
import numpy as np
import glob
from PIL import Image
import os





img_paths = glob.glob(f'{folder.rstrip("/")}/*.jpg') + glob.glob(f'{folder.rstrip("/")}/*.png')
for img_path in img_paths:
    # img_path = glob.glob(f'{folder.rstrip("/")}/*.jpg')[0]
    img = cv2.imread(img_path)
    undistorted = undistort(img, mtx, dist)
    #Image.fromarray(np.array(undistorted)).save("./" + img_path.split("/")[len(img_path.split("/")) -1].split("\\")[1][1:])
    
    Image.fromarray(np.array(undistorted)).save("./" + os.path.basename(img_path))