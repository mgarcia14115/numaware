
import sys
import os
while os.getcwd() != "/" and ".gitignore" not in os.listdir(os.getcwd()):
	os.chdir("..")
	if os.getcwd() == "/":
		print("COULD NOT FIND pyproject.toml.  Invalid project base file.")


print("Current Working Directory:  ", os.getcwd())

from data_collection.collector import Data_Collector
import cv2 as cv
import numpy as np
import glob
from PIL import Image

rows = int(sys.argv[1])
cols = int(sys.argv[2])
dir_pth = sys.argv[3]

obj = Data_Collector()

img_num = obj.get_img_count(dir_pth)

cap = cv.VideoCapture(-1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    isSucces, corners = cv.findChessboardCorners(gray, (rows, cols), flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FILTER_QUADS)
    
    if isSucces:

        isImageSaved = cv.imwrite(os.path.join(dir_pth,"img" + str(img_num) + ".png"),np.asarray(frame))
        if isImageSaved:
            print(f"Image {img_num} Saved Successfuly")
        img_num+=1
        
    
    
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()