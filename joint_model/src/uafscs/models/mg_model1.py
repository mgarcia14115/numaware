
import torch 


class Joint_predictor(torch.nn.Module):


    def __init__(self,dropout):
        
        super(Joint_predictor,self).__init__()

       
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  

        
        self.fc1 = torch.nn.Linear(307202, 128)#307202  
        self.fc2 = torch.nn.Linear(128, 6)  

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, midpoints):
        
        x = self.relu(self.conv1(x))
        x = self.pool1(x)  
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)  
        
        x = self.relu(self.conv3(x))
        x = self.pool3(x)  
       
        x = torch.flatten(x, start_dim=1) 
        x = torch.cat((x, midpoints), 1)  

      
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  
        
        return x



class Cart_predictor(torch.nn.Module):


    def __init__(self,dropout,output_layer):

        super(Cart_predictor,self).__init__()


        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)


        self.fc1 = torch.nn.Linear(307202, 128)#307202  
        self.fc2 = torch.nn.Linear(128, 3)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, midpoints):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, midpoints), 1)


        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

