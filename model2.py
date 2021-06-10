#!usr/bin/python
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

batch_size = 64
epochs = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
transform = transforms.Compose([transforms.Resize((30, 30)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
dataset = ImageFolder(root = '/run/media/raj/New Volume/MACHINE_LEARNING-DEEP_LEARNING/Traffic_Sign/data/train', transform = transform)
val_size = int(0.2*len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers=2)
validation_loader = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers=2)

class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(num_features = 32),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 5),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(num_features = 32),
            nn.MaxPool2d(kernel_size = 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(num_features = 64),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(num_features = 64),
            nn.MaxPool2d(kernel_size = 2),
            nn.Dropout2d(0.25)
        )
        self.fc_layers = nn.Sequential(nn.Linear(in_features = 64*3*3, out_features = 256),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(in_features = 256, out_features = 43),
            )
    def forward(self, image):
        output = self.conv_layers(image)
        output = output.view(-1, 64*3*3)
        output = self.fc_layers(output)
        return output

cnn = CNN_model().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.001)
loss_func = nn.CrossEntropyLoss()
for epoch in range(epochs):
    cnn.train()
    running_loss = 0.0
    for (image, label) in train_dataloader:
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = cnn(image)
        #print(output.size())
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch: {}/{}, loss is: {}'.format(epoch+1, epochs, running_loss/batch_size))
    
print('Training finished!')
cnn.eval()
accuracy = 0
count = 0
for i in validation_loader:
    images = i[0]
    labels = i[1]
    labels = np.array(labels)
    labels = torch.tensor(labels)
    labels = labels.to(device)
    #print(labels)
    images = images.to(device)
    labels = labels.to(device)
    out = cnn(images)
    #print('output is:',out)
    _, pred = torch.max(out, 1)
    #print(pred)
    accuracy += int(torch.sum(pred == labels))

accuracy = (accuracy/val_size)*100
print('Validation accuracy is: ', accuracy)
path = '/run/media/raj/New Volume/MACHINE_LEARNING-DEEP_LEARNING/Traffic_Sign/parameters.pth'
torch.save(cnn.state_dict(), path)



