import torch


class regression_with_midpoints_pose(torch.nn.Module):
    def __init__(self):
        super(regression_with_midpoints_pose, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3)
        )

    def forward(self, image, mid, **kwargs):
        return self.seq(mid)


class regression_with_images_midpoints(torch.nn.Module):
    def __init__(self):
        super(regression_with_images_midpoints, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, (3,3), 1, 1),
            torch.nn.MaxPool2d((2,2), 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, (3,3), 1, 1),
            torch.nn.MaxPool2d((2,2), 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, (3,3), 1, 1),
            # torch.nn.MaxPool2d((2,2), 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, (3,3), 1, 1),
            # torch.nn.MaxPool2d((2,2), 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, (3,3), 1, 1),
            torch.nn.MaxPool2d((2,2), 2, 1),
            torch.nn.ReLU(),
            
        )

        self.lin = torch.nn.Sequential(
            torch.nn.Linear(86018, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 3),
        )

    def forward(self, image, mid, **kwargs):
        x = self.conv(image)
        x = torch.flatten(x, start_dim=1) 
        x = torch.cat((x, mid), 1)
        x = self.lin(x)
        return x


