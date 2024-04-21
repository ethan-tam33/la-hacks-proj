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
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # Convert images to tensor
])

dataset = datasets.ImageFolder(root='data', transform = transformResizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

train_dataset = []
val_dataset = []


for sample in dataloader:
    x, y = sample
    x = x[0]
    if len(val_dataset) <= int(0.125 * len(dataset)):
        val_dataset.append((x, y[0]))
    else:
        train_dataset.append((x, y[0]))



# %%
# dataset = [(img, class), ...]
channeled_dataset = []

# %%


# %%
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# %%
epochs = 40
learning_rate = 0.00001
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

