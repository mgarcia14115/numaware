import sys
import os

while os.getcwd() != "/" and ".gitignore" not in os.listdir(os.getcwd()):
	os.chdir("..")
	if os.getcwd() == "/":
		print("COULD NOT FIND pyproject.toml.  Invalid project base file.")


print("Current Working Directory:  ", os.getcwd())

from ultralytics                       import YOLO
from data_collection.collector import Data_Collector
num_args = len(sys.argv)

if num_args < 2:
    print("please provide an image directory path and the path to the calibration files.")
    print("Example input: python collect.py img_dir_path cam_cali_path")
else:

    imgs_dir_path = sys.argv[1]
    cal_file_path = sys.argv[2]
    
    obj = Data_Collector(imgs_dir_path = imgs_dir_path , cam_calibration_path = cal_file_path)
    
    img_count = obj.get_img_count(imgs_dir_path)
    
    img_name  = "img" + str(img_count) + ".png"
    

    model = YOLO("../object_detection_model/model/mg_model_6/best.pt")


    response = "n"

    while(response == "n"):
        img_path = obj.take_image(img_name,2)
        model(source = img_path,conf = .6,show=True)
        print(f"Did all classes get predicted correctly? Enter [y|n]")
        response = input()
    
    results = model.predict(img_path,conf=.6)

    # Iterate through the results and print bounding box coordinates
    for r in results:
        boxes = r.boxes
      
        for box in boxes:
            print(f"\nclass for this box is: {box.cls}")
            # Get bounding box coordinates in (x1, y1, x2, y2) format
            xyxy = box.xyxy[0]
            print("Bounding Box Coordinates:", xyxy)
 
    