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
from matplotlib                         import pyplot as plt
from ultralytics                        import YOLO
from data_collection.collector          import Data_Collector
from data_collection.abb                import Robot

#[[264.04, 15.76, 640.51], [0.472, 0.498, 0.508, 0.52]]

# When facing closest shelf, change the first list second value (341.52).
# R.set_cartesian([[-81.38, 341.52, 335.7], [0.002, 0.039, -0.711, -0.702]])

try:
      R = Robot(ip="192.168.125.1")
except:
      print(f"Unable to connect!")
      exit()
      

#R.set_cartesian([[-62.76, 415.21, 173.48], [0.002, 0.039, -0.711, -0.702]])
#R.set_joints([32.8,32.49,9.95,-66.21,-68.98,132.6])
# [, back&front,]
#R.set_joints([32.8,32.49,9.95,-66.21,-68.98,132.6]) 
#R.set_cartesian([[100.44, 310.92, 182.91], [0.032, -0.005, 0.713, 0.701]])
try:
    #R.set_joints([32.8,32.49,9.95,-66.21,-68.98,132.6])
    print(f"Checking {str(R.get_cartesian())}")
    print(f"Checking {str(R.get_joints())}")
except:
    R.close()
    exit()

R.close()

