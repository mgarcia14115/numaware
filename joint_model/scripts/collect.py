import os
while os.getcwd() != "/" and ".gitignore" not in os.listdir(os.getcwd()):
	os.chdir("..")
	if os.getcwd() == "/":
		print("COULD NOT FIND gitignore.  Invalid project base file.")
print("Current Working Directory:  ", os.getcwd())
import  sys
from    PIL                             import Image
import matplotlib
matplotlib.use('TkAgg')
import cv2                              as cv
from matplotlib                         import pyplot as plt
from ultralytics                        import YOLO
from data_collection.collector          import Data_Collector
from data_collection.abb                import Robot






num_args = len(sys.argv)

if num_args < 4:
    print("\n\nPlease provide the command line args. \n")
    print("Example input: python collect.py img_dir_path data_file_path cam_idx\n\n")
else:

    imgs_dir_path  = sys.argv[1]      # arg holding the path to directory where images will be saved
    data_file_path = sys.argv[2]      # arg holding the path to data.csv file
    cam_idx        = int(sys.argv[3]) # arg holding the camera index
    
    
    obj = Data_Collector(imgs_dir_path = imgs_dir_path) # Creating a data collector object 
    img_count = obj.get_img_count(imgs_dir_path)        # Getting the count of how many images are in the folder
    img_name  = "img" + str(img_count) + ".png"         # Giving the image a name with its number count
    

    model = YOLO("../object_detection_model/model/ct_model_1/best.pt") # Loading in yolo model 




    # The loop will iterate until an image with all the correct classes are labeled 
    # This is done to verify the data we are saving is correctly labeled 
    # If the image is correctly labeled hit the letter y otherwise hit n
    
    response = "n"                                                      # Current response is no to initiate the while loop 
    while(response != "y"):                                             # Keep asking until response is yes
        img = obj.take_image(img_name,cam_idx)                          # Take an image
        model(source = img,conf = .6,show=True)                         # Make a prediction and display it
        print(f"Did all classes get predicted correctly? Enter [y|n]")  # ask user if it was correctly labeled
        response = input().lower()                                      # User response
    
    results = model.predict(img,conf=.6) # Let the model Predict

    
    
    # Iterate through the results and print bounding box coordinates
     

    all_joints      = [] # This list will store the joints for all predictions
    yolo_midpoints  = [] # All of midpoints 
    our_midpoints   = [] 
    all_carts
    

    for r in results:
        boxes = r.boxes
        idx = 0
        for box in boxes:
            cls = box.cls.item()
            
            # Get bounding box coordinates in (x1, y1, x2, y2) format
            
            xyxy = box.xyxy[idx]
            x_mid ,y_mid = obj.midpoint(xyxy)
            x_mid = round(x_mid,3)
            y_mid = round(y_mid,3)
            yolo_midpoints.append([str(x_mid) +"-" +str(y_mid)+":"+str(cls)])


            x2_mid = 0.0
            y2_mid = 0.0
            print(f"\n\n\n1. Click on the midpoint of the pallet. Exit the window when pressed.")
            
            def on_press(event):
                global x2_mid,y2_mid
                x2_mid,y2_mid = round(event.xdata,3),round(event.ydata,3)
                print('you pressed',x2_mid,y2_mid)
    
           
            fig = plt.figure()
            plt.imshow(img)
            plt.plot(x_mid,y_mid,marker=".",markersize=25)
            
            cid = fig.canvas.mpl_connect('button_press_event', on_press)
          
            plt.show()     

          
            print(f"2. Go and record the Joint positions for the {cls} pallet in manual mode: >>>>")
            print(f"When finished putting the robot in the desired spot and you are in programatic mode. Press Enter")
            input()

           
            try:
                R = Robot(ip='192.168.125.1')
            except:
                print(f"The robot is not in programatic mode.")
                exit()
            
                
            joints = obj.parse_joints(R.get_joints())
            print(f"The following Joints where recorded: {joints}\n\n\n")
            
            carts_pose = R.get_cartesian()
            carts = carts_pose[0]
            pose  = carts_pose[1]
            carts[1] = carts[1] - 60 # Move arm back

            
            # try:
            #     #R.set_cartesian([carts,pose])
            # except:
            #     print(f"Error moving cartesians back")
            #     exit()

            carts_pose = R.get_cartesian()
            carts = carts_pose[0]
            pose  = carts_pose[1]

            if carts[0] < -50: 
                carts[0] = carts[0] + 600 # Move arm left
            elif carts[0] < 90:
                carts[0] = carts[0] + 300 # Move arm left
            else:
                carts[0] = carts[0] + 100

            try:
                R.set_cartesian([carts,pose])
            except:
                print(f"Error moving cartesians left")
                exit()


            try:              
               R.set_cartesian([[364.01, 277, 364.08], [0.01, 0.009, -0.694, -0.72]]) # Move to starting location
            except:
                 print(f"Error moving joints")
                 exit()
            R.close()
            #########################################
            all_joints.append([joints+":"+str(cls)])
            our_midpoints.append([str(x2_mid) +"-" +str(y2_mid)+":"+str(cls)])

    cv.imwrite(os.path.join(imgs_dir_path,img_name),img)
    obj.save_data(data_file_path,img_name,all_joints,yolo_midpoints,our_midpoints)
