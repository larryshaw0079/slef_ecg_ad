import math

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True)
        )

        # Only conv2 degrades the scale
        self.conv2 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=11, stride=stride, padding=5, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(output_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        # If stride == 1, the length of the time dimension will not be changed
        # If input_channels == output_channels, the number of channels will not be changed
        # If the channels are mismatch, the conv1d is used to upgrade the channel
        # If the time dimensions are mismatch, the conv1d is used to downsample the scale
        self.downsample = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm1d(output_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        residual = self.downsample(x)  # Downsampe is an empty list if the size of inputs and outputs are same
        out += residual
        out = self.relu(out)

        return out


class ResidualBlockDilated(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super(ResidualBlockDilated, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=3, dilation=1, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True)
        )

        # Only conv2 degrades the scale
        self.conv2 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=3, dilation=2, stride=stride, padding=2,
                      bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=3, dilation=4, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(output_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        # If stride == 1, the length of the time dimension will not be changed
        # If input_channels == output_channels, the number of channels will not be changed
        # If the channels are mismatch, the conv1d is used to upgrade the channel
        # If the time dimensions are mismatch, the conv1d is used to downsample the scale
        self.downsample = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm1d(output_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)  # + self.conv12(x)
        out = self.conv2(out)  # + self.conv22(out)
        out = self.conv3(out)  # + self.conv32(out)

        residual = self.downsample(x)  # Downsampe is an empty list if the size of inputs and outputs are same
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, num_classes: int):
        super(ResNet, self).__init__()

        # The first convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual layers
        self.layer1 = self.__make_layer(ResidualBlockDilated, hidden_channels, hidden_channels, 2, stride=1)
        self.layer2 = self.__make_layer(ResidualBlockDilated, hidden_channels, hidden_channels * 2, 2, stride=2)
        self.layer3 = self.__make_layer(ResidualBlockDilated, hidden_channels * 2, hidden_channels * 4, 2, stride=2)
        self.layer4 = self.__make_layer(ResidualBlockDilated, hidden_channels * 4, hidden_channels * 8, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(
            1)  # Pooling operation computes the average of the last dimension (time dimension)

        # A dense layer for output
        self.fc = nn.Linear(hidden_channels * 8, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __make_layer(self, block, input_channels, output_channels, num_blocks, stride):
        """
        Get the residual layer
        @param block: The residual block
        @param input_channels: The number of input channels
        @param output_channels: The number of output channels
        @param num_blocks: The number of blocks in the layer
        @param stride: The stride of the convolution layer
        @return Torch.nn.Sequential
        """
        layers = []
        layers.append(block(input_channels, output_channels, stride=stride))
        for i in range(1, num_blocks):
            layers.append(block(output_channels, output_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        L_out = floor[(L_in + 2*padding - kernel) / stride + 1]
        @param x: (batch_size: 200, num_channels: 12, time_length: 5000)
        """
        out = self.conv1(x)  # (batch_size, num_channels: 64, time_length: 2500)
        out = self.layer1(out)  # (batch_size, num_channels: 64, time_length: 2500)
        out = self.layer2(out)  # (batch_size, num_channels: 128, time_length: 1250)
        out = self.layer3(out)  # (batch_size, num_channels: 256, time_length: 625)
        out = self.layer4(out)  # (batch_size, num_channels: 512, time_length: 313)

        out = self.avg_pool(out)  # (batch_size, num_channels: 512, time_length: 1)
        out = out.view(x.size(0), -1)  # (batch_size, num_channels: 512)
        out = self.fc(out)  # (batch_size, num_channels: 55)

        return out
