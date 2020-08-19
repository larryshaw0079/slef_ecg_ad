import math

import numpy as np

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_length, input_channel, num_class):
        super(Classifier, self).__init__()
        self.__dict__.update(locals())

        self.layer_num = int(np.log2(input_length)) - 3
        self.max_channel_num = input_length * 2
        self.final_length = 4

        self.conv_list = []

        for i in range(self.layer_num + 1):
            current_out_channel = self.max_channel_num // 2 ** (self.layer_num - i)

            if i == 0:
                self.conv_list.append(nn.Conv1d(in_channels=self.input_channel, out_channels=current_out_channel,
                                                kernel_size=4, stride=2, padding=1))
            else:
                self.conv_list.append(nn.Conv1d(in_channels=prev_channel, out_channels=current_out_channel,
                                                kernel_size=4, stride=2, padding=1))
                self.conv_list.append(nn.BatchNorm1d(current_out_channel))
            self.conv_list.append(nn.LeakyReLU(0.2, inplace=True))
            prev_channel = current_out_channel

        self.conv_layers = nn.Sequential(*self.conv_list)

        # Pooling operation computes the average of the last dimension (time dimension)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # A dense layer for output
        self.fc = nn.Linear(self.max_channel_num, num_class)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_layers(x)
        out = self.avg_pool(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)

        return out
