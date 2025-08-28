import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,dropout, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x += shortcut
        x = self.relu(x)
        return x

class Conv2DRegressionModel(nn.Module):
    def __init__(self,dropout = 0.3):
        super(Conv2DRegressionModel, self).__init__()
        
        self.res_block1 = ResidualBlock(in_channels=1, out_channels=32, dropout=dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_block2 = ResidualBlock(in_channels=32, out_channels=64, dropout=dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_block3 = ResidualBlock(in_channels=64, out_channels=128, dropout=dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block4 = ResidualBlock(in_channels=128, out_channels=256, dropout=dropout)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block5 = ResidualBlock(in_channels=256, out_channels=512, dropout=dropout)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(512, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn_fc3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.bn_fc4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, 1)


        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.res_block1(x)
        x = self.pool1(x)
        
        x = self.res_block2(x)
        x = self.pool2(x)

        x = self.res_block3(x)
        x = self.pool3(x)

        x = self.res_block4(x)
        x = self.pool4(x)

        x = self.res_block5(x)
        x = self.pool5(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_fc4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        
        return x
