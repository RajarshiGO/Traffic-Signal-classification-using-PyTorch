import torch.nn as nn
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

