import sys

from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import os
while os.getcwd() != "/" and ".gitignore" not in os.listdir(os.getcwd()):
	os.chdir("..")
	if os.getcwd() == "/":
		print("COULD NOT FIND pyproject.toml.  Invalid project base file.")


print("Current Working Directory:  ", os.getcwd())

from ultralytics                       import YOLO
from data_collection.collector         import Data_Collector
from src.abb                           import Robot
num_args = len(sys.argv)

if num_args < 4:
    print("please provide an image directory path and the path to the calibration files.")
    print("Example input: python collect.py img_dir_path data_file_path cam_cali_path cam_idx")
else:

    imgs_dir_path  = sys.argv[1]
    data_file_path = sys.argv[2]
    cal_file_path  = sys.argv[3]
    cam_idx        = int(sys.argv[4])
    
    
    obj = Data_Collector(imgs_dir_path = imgs_dir_path , cam_calibration_path = cal_file_path)
    
    img_count = obj.get_img_count(imgs_dir_path)
    
    img_name  = "img" + str(img_count) + ".png"
    

    model = YOLO("../object_detection_model/model/mg_model_6/best.pt")


    response = "n"

    while(response == "n"):
        img_path = obj.take_image(img_name,cam_idx)
        model(source = img_path,conf = .6,show=True)
        print(f"Did all classes get predicted correctly? Enter [y|n]")
        response = input().lower()
    
    results = model.predict(img_path,conf=.6)

    # Iterate through the results and print bounding box coordinates
    all_joints = []
    midpoints = []
    for r in results:
        boxes = r.boxes
      
        for box in boxes:
            cls = box.cls.item()
            print(f"\nclass for this box is: {cls}")
            # Get bounding box coordinates in (x1, y1, x2, y2) format
            
            xyxy = box.xyxy[0]
            x_mid ,y_mid = obj.midpoint(xyxy)
            x_mid = round(x_mid,3)
            y_mid = round(y_mid,3)
            print("Bounding Box Coordinates:", xyxy)
            print(f"midpoint for bouding box: {x_mid},{y_mid}")            

            print("Go and record Joint positions for this pallet: >>>>")

    
            img = Image.open(img_path)
            plt.imshow(img)
            plt.plot(x_mid,y_mid,marker=".",markersize=25)
            plt.show()     
    
            print(f"When the robot is ready to connect. Press any key ")
            input()

            connected = False
            R = None
            while connected == False:
                
                try:
                    R = Robot(ip = '192.168.125.1')
                    connected = True
                except:
                        print(f"Error connecting")
            
            
            joints = obj.parse_joints(R.get_joints())
            R.close()
            all_joints.append([joints+":"+str(cls)])
            midpoints.append([str(x_mid) +"-" +str(y_mid)+":"+str(cls)])

    
    obj.save_data(data_file_path,img_name,all_joints,midpoints)