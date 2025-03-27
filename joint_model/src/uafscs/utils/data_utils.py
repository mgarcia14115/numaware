import pandas as pd
import torch
import torchvision as tv
import os



def strList_to_floatList(series,isCarts):
    new_list = []
    if isCarts!=True:
        for rec in series:
            new_rec = rec[1:-1]
            new_list.append([float(i.strip()) for i in new_rec.split(",")])

        return new_list
    else:
         for rec in series:
            all_rec = rec[1:-1].split(",")
            x = all_rec[0][1:].strip()
            y = all_rec[1].strip()
            z = all_rec[2][0:-1].strip()
           
            new_list.append([float(x),float(y),float(z)])
            
         return new_list






    

class UADataset(torch.utils.data.Dataset):

    def __init__(self,imgs_pth,csv_file):
     
        df = pd.read_csv(csv_file)
       
        self.imgs_pth       = imgs_pth
        self.joints         = torch.tensor(strList_to_floatList(df["joints"],False))
        self.carts          = torch.tensor(strList_to_floatList(df["cartesians"],True))
        self.yolo_midpoints = torch.tensor(strList_to_floatList(df["yolo_midpoint"],False))
        self.our_midpoints  = torch.tensor(strList_to_floatList(df["our_midpoint"],False))
        self.img_names      = df["img"]
        
    def __len__(self):

        return len(self.joints)
    
    def __getitem__(self,idx):
  
        joints        = self.joints[idx]
        carts         = self.carts[idx]
        yolo_midpoint = self.yolo_midpoints[idx]
        our_midpoint  = self.our_midpoints[idx]
        img           = tv.io.read_image(os.path.join(self.imgs_pth,self.img_names[idx]))
        img           = tv.transforms.Resize((480,640))(img)
        img           = img/255
     
        return img,yolo_midpoint,our_midpoint,joints,carts