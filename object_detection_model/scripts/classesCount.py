import os 
import sys

path = sys.argv[1]


files = os.listdir(path)

w_count = 0
b_count = 0
g_count = 0

for file in files:
    f = open(path + "/" + file, "r")
    for line in f:
        cls = int(line[0])
        if cls == 0:
            g_count+=1
        elif cls == 1:
            w_count +=1
        else:
            b_count +=1



