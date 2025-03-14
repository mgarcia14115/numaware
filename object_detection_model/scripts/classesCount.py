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
        cls = -1
        try:
            cls = int(line[0])
        except:
            print(f"The first character is not a number")
        if cls == 0:
            g_count+=1
        elif cls == 1:
            w_count +=1
        elif cls == 2:
            b_count +=1
        else:
            print("Not a class")


print(f"There is {str(g_count)} grey pallets\n")
print(f"There is {str(w_count)} white pallets\n")
print(f"There is {str(b_count)} black pallets\n")

