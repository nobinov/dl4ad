from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os

import cv2
import numpy as np

torch.manual_seed(1)
device = torch.device("cuda")

# loading the data

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_transforms2 = transforms.Compose([
							transforms.Resize(32), 
							transforms.ToTensor(),
							transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
							])

data_dir = '../data/GTSRB'
image_datasets = datasets.ImageFolder('../data/GTSRB/Final_Training/Images', data_transforms2) 
print(image_datasets)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=4, shuffle=True, num_workers=4)
print(dataloaders)
dataset_sizes = len(image_datasets)
print(dataset_sizes)
class_names = image_datasets.classes
print(class_names)
#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                          data_transforms[x])
#                  for x in ['Final_Training/Images', 'Final_Test/Images']}
#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
#                                             shuffle=True, num_workers=4)
#              for x in ['Final_Training/Images', 'Final_Test/Images']}
#dataset_sizes = {x: len(image_datasets[x]) for x in ['Final_Training/Images', 'Final_Test/Images']}
#class_names = image_datasets['Final_Training/Images'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
#inputs, classes = next(iter(dataloaders))
#print(enumerate(dataloaders))

# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])

# get some random training images
dataiter = iter(dataloaders)
print(next(dataiter))
#images, labels = dataiter.next()

print('------')
#print(dataiter, images, labels)

# show images
#imshow(torchvision.utils.make_grid(images))