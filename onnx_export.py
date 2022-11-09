import io
import os

import numpy as np
from torch import nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

import torch.onnx

file_name = "resnet50_20ep.onnx"
epochs = 20

model = models.resnet50(pretrained=True)

num_class = 10
fc_in = model.fc.in_features
model.fc = nn.Linear(fc_in, num_class)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), 0.1,
                            momentum=0.9,
                            weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='/home/esoc/datasets/cifar10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


def train(train_loader, model, criterion, optimizer, epoch, device):
    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
for epoch in range(epochs):
    # train for one epoch
    print('Training epoch num: %d' % (epoch))  
    train(trainloader, model, criterion, optimizer, epoch, device)
    
    scheduler.step()
    

"""
Export to ONNX
"""
model.eval()

x = torch.randn(batch_size, 3, 32, 32, requires_grad=True)
x = x.cuda()

#export from pytorch to onnx
torch.onnx.export(model,
                  x,
                  file_name,
                  export_params = True,
                  input_names = ['input'],
                  output_names = ['output'],
                  dynamic_axes = {'input' : {0 : 'batch_size'},   
                                'output' : {0 : 'batch_size'}}
                  )
