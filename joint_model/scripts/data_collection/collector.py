import os
while os.getcwd() != "/" and ".gitignore" not in os.listdir(os.getcwd()):
	os.chdir("..")
	if os.getcwd() == "/":
		print("COULD NOT FIND pyproject.toml.  Invalid project base file.")


print("Current Working Directory:  ", os.getcwd())
import cv2     as cv
import numpy   as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import data_collection.calibri as cal
import matplotlib.pyplot as plt




class Data_Collector:

    def __init__(self , imgs_dir_path = None):
        self.imgs_dir_path = imgs_dir_path
     
    
    def take_image(self, img_name , cam_idx):

        abs_path = os.path.join(self.imgs_dir_path,img_name)

        cap = cv.VideoCapture(cam_idx)
        
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        ret, frame = cap.read()

        if not ret:
            print("Can't take an image. Exiting ....")
            cap.release()
            cv.destroyAllWindows()
            return None
        else:       
            
            success = cv.imwrite(abs_path,frame)
            
            if success:
                print(f"Image successfully saved to {abs_path}")
               
            else:
                print(f"An error occured. The image was not able to be captured.")
                             
        cap.release()
        cv.destroyAllWindows()
        return abs_path

    
    def get_img_count(self, filepath):
        
        img_dir = os.listdir(filepath)
        
        count = 0

        for file in img_dir:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                count+=1

        return count

    def find_camera_idx(self):
        
        for cam_idx in range(-1,6):

            cap = cv.VideoCapture(cam_idx)
            
            if cap.isOpened():
                print(f"Camera is open on idx {cam_idx}")
    

    def midpoint(self, coordinates):

        x1 = coordinates[0].item()
        y1 = coordinates[1].item()

        x2 = coordinates[2].item()
        y2 = coordinates[3].item()

        return ((x1+x2)/2),((y1+y2)/2)
    
    def save_data(self, file_csv, img_name, all_joints, midpoints):
        
        df = pd.read_csv(file_csv)

        try:
            idx = df.index[-1] + 1
        except:
            idx = 0

        file = "images/" + img_name

        df.loc[idx] = [file,all_joints,midpoints]

        df.to_csv(file_csv,index=False)

    def parse_joints(self,joints):

        parsed_joints = ""

        for j in joints:
            parsed_joints += str(j)+"-"

        return parsed_joints[0:-1]
    


     
    
        


        


