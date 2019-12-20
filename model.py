import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class QNetwork(nn.Module):

    def __init__(self, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding = 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*5*5, 64)
        self.fc1_bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,action_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)        
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)        
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
        # print(x.shape)
        #x = F.relu(self.fc1_bn(self.fc1(x.flatten(1))))
        # print(x.shape)
        return self.fc1(x.flatten(1))
