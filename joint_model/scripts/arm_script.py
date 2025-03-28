import cv2
import torch
from ultralytics import YOLO
import torchvision.transforms as transforms
import abb

# Capturing image
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Camera not detected")

    ret, image = cap.read()
    cv2.imwrite('test_image.jpg', image)

    cap.release()
except:
    print("Error taking/saving image. Try again.")

# Method for grabbing midpoints
def grab_midpoints(bounding_box):
    x1 = bounding_box[0][0]
    y1 = bounding_box[0][1]
    x2 = bounding_box[0][2]
    y2 = bounding_box[0][3]

    midpoint_x = (x1+x2)/2
    midpoint_y = (y1+y2)/2

    return midpoint_x, midpoint_y

# Grabbing yolo midpoints and classes
yolo_model = YOLO("../../object_detection_model/model/ct_model_1/best.pt")

results = yolo_model('test_image.jpg')

yolo_data = {'Black': 'None', 'Grey': 'None', 'White': 'None'}

for result in results[0]:
    # Detected gray
    if int(result.boxes.cls) == 0:
        # Grabing midpoint
        bounding_box = result.boxes.xyxy.tolist()
        
        midpoint_x, midpoint_y = grab_midpoints(bounding_box)
        
        yolo_data['Grey'] = (midpoint_x, midpoint_y)

    # Detected white
    if int(result.boxes.cls) == 1:
        # Grabing midpoint
        bounding_box = result.boxes.xyxy.tolist()
        
        midpoint_x, midpoint_y = grab_midpoints(bounding_box)
        
        yolo_data['White'] = (midpoint_x, midpoint_y)

    # Detected black
    if int(result.boxes.cls) == 2:
        # Grabing midpoint
        bounding_box = result.boxes.xyxy.tolist()
        
        midpoint_x, midpoint_y = grab_midpoints(bounding_box)
        
        yolo_data['Black'] = (midpoint_x, midpoint_y)

# Test model
class JMRIM(torch.nn.Module):

    def __init__(self, dropout):

        super(JMRIM,self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(153602, 256)
        self.fc2 = torch.nn.Linear(256, 3)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, x, midpoints):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.relu(self.conv4(x))
        x = self.pool4(x)

        x = torch.flatten(x, start_dim=0)
        x = x.unsqueeze(0)
        x = torch.cat((x, midpoints), 1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = JMRIM(0.0)
model.load_state_dict(torch.load("../local/jmrim.pt"))

model = model.to(device)
model.eval()

selected_class = input("What class would you like?")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640,480))
])



with torch.no_grad():

    image = transform(image).to(device)

    if selected_class.lower() == 'black':
        
        midpoint = torch.tensor(yolo_data['Black']).unsqueeze(0).to(device)

        predictions = model(image, midpoint)
    if selected_class.lower() == 'grey':

        midpoint = torch.tensor(yolo_data['Grey']).unsqueeze(0).to(device)

        predictions = model(image, midpoint)        
    if selected_class.lower() == 'white':

        midpoint = torch.tensor(yolo_data['White']).unsqueeze(0).to(device)

        predictions = model(image, midpoint)


predictions = predictions.tolist()

R = abb.Robot(ip='192.168.125.1')

R.set_cartesian([predictions[0],[0.01, 0.009, -0.694, -0.72]])

R.close()

