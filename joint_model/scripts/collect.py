import os
while os.getcwd() != "/" and ".gitignore" not in os.listdir(os.getcwd()):
	os.chdir("..")
	if os.getcwd() == "/":
		print("COULD NOT FIND pyproject.toml.  Invalid project base file.")
print("Current Working Directory:  ", os.getcwd())







import  sys
from    PIL                             import Image
import matplotlib
matplotlib.use('TkAgg')
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
    

    model = YOLO("../object_detection_model/model/mg_model_17/best.pt") # Loading in yolo model 




    # The loop will iterate until an image with all the correct classes are labeled. 
    # This is done to verify the data we are saving is correctly labeled. 
    # If the image is correctly saved hit the letter y.
    
    response = "n"                                                      # Current response is no to initiate the while loop 
    while(response != "y"):                                             # Keep asking until response is yes
        img_path = obj.take_image(img_name,cam_idx)                     # Take an image
        model(source = img_path,conf = .6,show=True)                    # Make a prediction and display it
        print(f"Did all classes get predicted correctly? Enter [y|n]")  # ask user if it was correctly labeled
        response = input().lower() # User response
    
    results = model.predict(img_path,conf=.6)

    # Iterate through the results and print bounding box coordinates
    all_joints = []
    midpoints  = []
    
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
           
            print(f"\n\n\n1. Click on the midpoint of the pallet. Exit the window when pressed.")
            
            def on_press(event):
                global x_mid,y_mid
                x_mid,y_mid = event.xdata,event.ydata
                print('you pressed',x_mid,y_mid)
    
            img = Image.open(img_path)
            fig = plt.figure()
            plt.imshow(img)
            plt.plot(x_mid,y_mid,marker=".",markersize=25)
            
            cid = fig.canvas.mpl_connect('button_press_event', on_press)
          
            plt.show()     

            #change this to work with the robot api
            print(f"2. Go and record the Joint positions for this pallet in manual mode: >>>>")
            print(f"3.Press enter when the robot is in programatic mode.")
            input()
            
            try:
                R = Robot(ip='192.168.125.1')
            except:
                print(f"The robot is not in programatic mode.")
                exit()
            

            joints = obj.parse_joints(R.get_joints())
            print(f"The following Joints where recorded: {joints}\n\n\n")
            R.close()
            #########################################
            all_joints.append([joints+":"+str(cls)])
            midpoints.append([str(x_mid) +"-" +str(y_mid)+":"+str(cls)])

    
    obj.save_data(data_file_path,img_name,all_joints,midpoints)
