import torch
from torch import nn

import torch
import torch.nn.functional as F

class CustomPad(nn.Module):
    def __init__(self, padding):
        super(CustomPad, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding)

class DRPDetNet(nn.Module):
 
    def __init__(self):
        super(DRPDetNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
            CustomPad(padding=(0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=(0, 1)),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            # nn.ReLU()  # You might replace this with your custom RegressionMSELayer
        )

    def forward(self,x):
        x = self.conv_layers(x)
        return x
    

class DRPClsNet(nn.Module):
    def __init__(self, num_class):
        super(DRPClsNet, self).__init__()
        self.convLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=11, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=1),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=1),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 57 * 57, 128)
        self.fc2 = nn.Linear(128, num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.convLayers(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
