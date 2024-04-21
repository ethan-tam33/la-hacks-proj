# %% [markdown]
# Install Needed Packages
# 

# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
from pathlib import Path
from tqdm import tqdm
import pickle


# %%
import torch.nn.functional as F

# %%
#import tqdm.notebook as tqdm


# %%
import torch.nn as nn
import numpy as np

# %%
print("Cuda available: ", torch.cuda.is_available())
if(torch.cuda.is_available()):
    torch.cuda.set_device("cuda:0")
    print("Is cuDNN version:", torch.backends.cudnn.version())
    print("cuDNN enabled:a", torch.backends.cudnn.enabled)
    print("Device count: ", torch.cuda.device_count())
    print("Current device: ", torch.cuda.current_device())
    print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
#Setup device agnostic code (i.e use GPU if possible)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# %% [markdown]
# Process the Data + Create a Dataloader

# %%

transformResizer = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),  # Convert images to tensor
    transforms.RandomAffine(15),
])

dataset = datasets.ImageFolder(root='data', transform = transformResizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

train_dataset = []
val_dataset = []


for sample in dataloader:
    x, y = sample
    x = x[0]
    if len(val_dataset) <= int(0.10 * len(dataset)):
        val_dataset.append((x, y[0]))
    else:
        train_dataset.append((x, y[0]))



# %%
# dataset = [(img, class), ...]
channeled_dataset = []

# %%
# Residual Block with Dropout
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.2):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Apply dropout after the first convolution
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet Model with Dropout
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, dropout_rate=0.4):
        super(ResNet, self).__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=3, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 1024, num_blocks[3], stride=2, dropout_rate=dropout_rate)
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc = nn.Linear(16384 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Instantiate the ResNet18 model
net = ResNet(BasicBlock, [8, 4, 2, 1], num_classes=2, dropout_rate=0.5).to(device)

# %%
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# %%
epochs = 40
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

net.train()  # Put model in training mode
for epoch in range(epochs):
    training_losses = []
    for x, y in tqdm(train_dataloader):
        x, y = x.float().to(device), y.type(torch.LongTensor).to(device)  # Change y to LongTensor
        optimizer.zero_grad()  # Remove the gradients from the previous step
        pred = net(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())
    net.train()
    print("Finished Epoch", epoch + 1, ", training loss:", np.mean(training_losses))
    #train_loss_arr.append(np.mean(training_losses))
    # determine train and validation accuracies
    with torch.no_grad():
      net.eval()  # Put model in eval mode
      num_correct_train = 0
      for x, y in train_dataloader:
          x, y = x.float().to(device), y.long().to(device)  # Change y to LongTensor
          pred = net(x)
          num_correct_train += torch.sum(torch.argmax(pred, dim=1) == y).item()  # Compare with argmax
      train_acc = num_correct_train / len(train_dataset)
      #train_acc_arr.append(train_acc)
      print("Train Accuracy:", train_acc)
      num_correct_val = 0
      for x, y in val_dataloader:
          x, y = x.float().to(device), y.long().to(device)  # Change y to LongTensor
          pred = net(x)
          num_correct_val += torch.sum(torch.argmax(pred, dim=1) == y).item()  # Compare with argmax
      val_acc = num_correct_val / len(val_dataset)
      #val_acc_arr.append(val_acc)
      print("Val Accuracy:", val_acc)
    net.train()  # Put model back in train mode

# save the reef classification model as a pickle file
model_pkl_file = "reef_classification_model.pkl"
with open(model_pkl_file, 'wb') as file:
    pickle.dump(net, file)

