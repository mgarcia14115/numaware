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

R = Robot(ip="192.168.125.1")

R.set_cartesian([[-62.76, 415.21, 173.48], [0.002, 0.039, -0.711, -0.702]])

try:
    print(f"Checking {str(R.get_cartesian())}")
except:
    R.close()
    exit()

R.close()

