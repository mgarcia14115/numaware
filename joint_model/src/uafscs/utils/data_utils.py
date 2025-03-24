from torch.utils.data import Dataset
import pandas as pd
import torch
import torchvision as tv
import os





class joint_dataset(Dataset):

    def __init__(self,imgs_pth,csv_file):
        
        df = pd.read_csv(csv_file)
        
        self.imgs_pth       = imgs_pth
        self.joints         = torch.tensor(df["joints"])
        self.yolo_midpoints = torch.tensor(df["yolo_midpoint"])
        self.our_midpoints  = torch.tensor(df["our_midpoint"])
        self.img_names      = df["img"]
        
    def __len__(self):

        return len(self.joints)
    
    def __get_item__(self,idx):

        joint         = self.joints[idx]
        yolo_midpoint = self.yolo_midpoints[idx]
        our_midpoint  = self.our_midpoints[idx]
        img           = tv.io.read_image(os.path.join(self.imgs_pth,self.img_names[idx]))
        img           = tv.transforms.Resize((640,480))(img)

        return img,yolo_midpoint,our_midpoint,joint





class cartesian_dataset(Dataset):
    def __init__(self,imgs_pth,csv_file):
        
        df = pd.read_csv(csv_file)
        
        self.imgs_pth       = imgs_pth
        self.carts          = torch.tensor(df["cartesians"])
        self.yolo_midpoints = torch.tensor(df["yolo_midpoint"])
        self.our_midpoints  = torch.tensor(df["our_midpoint"])
        self.img_names      = df["img"]
        
    def __len__(self):

        return len(self.carts)
    
    def __get_item__(self,idx):

        carts         = self.carts[idx]
        yolo_midpoint = self.yolo_midpoints[idx]
        our_midpoint  = self.our_midpoints[idx]
        img           = tv.io.read_image(os.path.join(self.imgs_pth,self.img_names[idx]))
        img           = tv.transforms.Resize((640,480))(img)

        return img,yolo_midpoint,our_midpoint,carts