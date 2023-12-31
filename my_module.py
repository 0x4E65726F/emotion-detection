import os
import numpy as np
import pandas as pd
import cv2

from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, img_channel, img_width, img_height, img_label, conv_channels, fc_channels, kernel_size = 5, padding = 0, stride = 1):
        super().__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = nn.ModuleList()
        self.fc = nn.ModuleList()
        
        cur_channel = img_channel
        cur_width = img_width
        cur_height = img_height
        for conv_channel in conv_channels:
            cur_width = ((cur_width - kernel_size + 2 * padding) // stride + 1) // 2
            cur_height = ((cur_height - kernel_size + 2 * padding) // stride + 1) // 2
            self.conv.append(nn.Conv2d(cur_channel, conv_channel, kernel_size, stride = stride, padding = padding))
            cur_channel = conv_channel
            
        cur_channel = cur_channel * cur_width * cur_height
        for fc_channel in fc_channels:
            self.fc.append(nn.Linear(cur_channel, fc_channel))
            cur_channel = fc_channel
        self.fc.append(nn.Linear(cur_channel, img_label))

    def forward(self,x):
        for conv in self.conv:
            x = self.pool(F.relu(conv(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        for i in range(len(self.fc) - 1):
            x = F.relu(self.fc[i](x))
            
        x = F.softmax(self.fc[len(self.fc) - 1](x), dim=1)
        return x

class Net2(nn.Module):
    def __init__(self, img_channel, img_width, img_height, img_label, conv_channels, fc_channels, kernel_size = 5, stride = 1):
        super().__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = nn.ModuleList()
        self.fc = nn.ModuleList()
        
        cur_channel = img_channel
        cur_width = img_width
        cur_height = img_height
        for conv_channel in conv_channels:
            padding = (conv_channel - 1) // 2
            cur_width = ((cur_width - kernel_size + 2 * padding) // stride + 1) // 2
            cur_height = ((cur_height - kernel_size + 2 * padding) // stride + 1) // 2
            self.conv.append(nn.Conv2d(cur_channel, conv_channel, kernel_size, stride = stride, padding = padding))
            cur_channel = conv_channel
            
        cur_channel = cur_channel * cur_width * cur_height
        for fc_channel in fc_channels:
            self.fc.append(nn.Linear(cur_channel, fc_channel))
            cur_channel = fc_channel
        self.fc.append(nn.Linear(cur_channel, img_label))

        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        for conv in self.conv:
            x = self.pool(F.relu(conv(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        for i in range(len(self.fc) - 1):
            x = self.dropout(F.relu(self.fc[i](x)))
            
        x = F.softmax(self.fc[len(self.fc) - 1](x), dim=1)
        return x

def read_pic(data, categories, limit = -1):
    folder_path = 'images/'
    image_paths = data.iloc[:, 1]
    labels = data.iloc[:, 2].str.lower()
    X = []
    label_to_int = {category: i for i, category in enumerate(categories)}
    integer_labels = labels.map(label_to_int)
    y = torch.LongTensor(integer_labels.to_numpy())
    count = 0
    for filename in image_paths:
        if count == limit:
            break
        count += 1
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for common image file extensions
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (350, 350))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                X.append(torch.Tensor(img))
    X = torch.stack(X).unsqueeze(1) # convert to pytorch tensor (batch, channels, w, h)
    if limit > 1:
        y = y[:limit]
    return X, y

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def get_loader(Xtr, Xva, ytr, yva, batch_size = 1):
    train_dataset = torch.utils.data.TensorDataset(Xtr, ytr)
    test_dataset = torch.utils.data.TensorDataset(Xva, yva)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def train_process(myDevice, myCNN, loader, LR = 0.001, EPOCH = 2, PRINT = 100):
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    print(criterion)
    optimizer = optim.SGD(myCNN.parameters(), lr=LR, momentum=0.9)
    optimizer_to(optimizer, myDevice)
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data[0].to(myDevice), data[1].to(myDevice)
            optimizer.zero_grad()
            outputs = myCNN(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % PRINT == PRINT - 1:    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT:.3f}')
                running_loss = 0.0
    print('Finished Training')
    return myCNN

def train_process2(myDevice, myCNN, loader, LR = 0.001, EPOCH = 2, PRINT = 100):
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    print(criterion)
    optimizer = optim.AdamW(myCNN.parameters(), lr=LR)
    optimizer_to(optimizer, myDevice)
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data[0].to(myDevice), data[1].to(myDevice)
            optimizer.zero_grad()
            outputs = myCNN(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % PRINT == PRINT - 1:    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT:.3f}')
                running_loss = 0.0
    print('Finished Training')
    return myCNN

def accuracy(myDevice, myCNN, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(myDevice), data[1].to(myDevice)
            # calculate outputs by running images through the network
            outputs = myCNN(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total)