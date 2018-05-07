from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#params
split_ratio = 0.2 # how much data that will be used as validation data
random_seed = 123 #seed for data shuffling

#---data loading----------------------------------------------
#---training
data_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
image_dataset = datasets.ImageFolder(root='../../../data/GTSRB/Final_Training/Images',
                                           transform=data_transform)
dataset_sizes = len(image_dataset)
print('dataset size : ' + str(dataset_sizes))
class_names = image_dataset.classes
indices = list(range(dataset_sizes))
split_size = int(np.floor(split_ratio * dataset_sizes))
print('split index : ' + str(split_size))

#randomizer setting
np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split_size:], indices[:split_size]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
print('amount of training data : ' + str(len(train_sampler)))
print('amount of validation data : ' + str(len(valid_sampler)))

train_loader = torch.utils.data.DataLoader(image_dataset, 
                                            batch_size=64, sampler=train_sampler,
                                            num_workers=4, pin_memory=True)

valid_loader = torch.utils.data.DataLoader(image_dataset, 
                                            batch_size=64, sampler=valid_sampler,
                                            num_workers=4, pin_memory=True)



#dataset_loader = torch.utils.data.DataLoader(image_dataset,
#                                             batch_size=64, shuffle=True,
#                                             num_workers=4)


print('train batch : ' + str(len(train_loader)))
print('valid batch : ' + str(len(valid_loader)))
#---divide into trainng and validation



#---test
#image_dataset_test = datasets.ImageFolder(root='../../../data/GTSRB/Final_Test/Images',
#                                           transform=data_transform)
#dataset_loader_test = torch.utils.data.DataLoader(image_dataset_test,
#                                             batch_size=64, shuffle=True,
#                                             num_workers=4)

#------------------------------------------------------------

#---try to show the image------------------------------------
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
    plt.pause(2)  # pause a bit so that plots are updated


# Get a batch of training data
#inputs, classes = next(iter(dataset_loader))

# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])
#---------------------------------------------------------------

#---construct the network---------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)

        self.linear1 = nn.Linear(64 * 4 * 4, 512)
        self.linear2 = nn.Linear(512, 43)

    def forward(self, x):

        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.dropout(x)
        x = F.log_softmax(self.linear2(x), dim=1)

        return x

# We create the network, shift it on the GPU and define a optimizer on its parameters
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
#---------------------------------------------------------------

#---training function-------------------------------------------
# This function trains the neural network for one epoch
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move the input and target data on the GPU
        data, target = data.to(device), target.to(device)
        # Zero out gradients from previous step
        optimizer.zero_grad()
        # Forward pass of the neural net
        output = model(data)
        # Calculation of the loss function
        loss = F.nll_loss(output, target)
        # Backward pass (gradient computation)
        loss.backward()
        # Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#---testing function--------------------------------------------
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            # if not pred.eq(target.view_as(pred)):   ## If you just want so see the failing examples
            #cv_mat = data.cpu().data.squeeze().numpy()
            #cv_mat = cv2.resize(cv_mat, (400, 400))
            #cv2.imshow("test image", cv_mat)
            #print("Target label is : %d" % target.cpu().item())
            #print("Predicted label is : %d" % (pred.cpu().data.item()))
            #cv2.waitKey()

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(valid_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
#--------------------------------------------------------------

#num_train_epochs = 1
#for epoch in range(1, num_train_epochs + 1):
#    train(epoch)

#test()