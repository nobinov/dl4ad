from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time

import cv2
import numpy as np

torch.manual_seed(1)
device = torch.device("cuda")


#####################################################
## This is the example code of the pytorch tutorial
## of the deep learning lab. Ich you want to
## visualize the inputs, uncomment the lines 94-99.
## Data will be automatically downloaded to ../data/
## Author: Johan Vertens
#####################################################


# Here we use predefined datasets from 'torchvision'
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True, pin_memory=True)

# Definition of the network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)    # FYI: In the lecture I forgot to add the padding, thats why the feature size calculation was wrong
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(980, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 980)   # Flatten data for fully connected layer. Input size is 28*28, we have 2 pooling layers so we pool the spatial size down to 7*7. With 20 feature maps as the output of the previous conv we have in total 7x7x20 = 980 features.
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# We create the network, shift it on the GPU and define a optimizer on its parameters
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

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

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
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

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

num_train_epochs = 10
for epoch in range(1, num_train_epochs + 1):
    train(epoch)

test()

