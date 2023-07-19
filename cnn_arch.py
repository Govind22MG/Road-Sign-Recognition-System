import torch
import torch.nn as nn
import torch.nn.functional as f

class CNN(nn.Module):
    def __init__(self, in_c, out_c):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 16, kernel_size = (3, 3), stride = 1, bias = False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = (3, 3), stride = 1, bias = False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = (3, 3), stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.fc1 = nn.Linear(8*8*64, 200)
        self.fc2 = nn.Linear(200, out_c)

    def forward(self, x):
        y = f.relu(self.conv1(x))
        y = self.maxpool(y)                
        y = f.relu(self.conv2(y))
        y = self.maxpool(y)                
        y = f.relu(self.bn1(self.conv3(y)))
        y = self.maxpool(y)                
        y = y.view(-1, 8*8*64)
        y = f.relu(self.fc1(y))
        y = f.sigmoid(self.fc2(y))
        return(y)
