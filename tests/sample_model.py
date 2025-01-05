import torch
import torch.nn as nn

class SAMPLE_MODEL(nn.Module):
    def __init__(self, input_size, input_channels, output_size, num_classes, hyperparams):
        self.input_size = input_size
        self.input_channels = input_channels
        self.output_size = output_size
        self.num_classes = num_classes
        self.hyperparams = hyperparams

        super().__init__()
        self.l0 = nn.Linear(input_size,10000)
        self.l1 = nn.Linear(10000,100)
        self.l2 = nn.Linear(100,output_size)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.l0(x)
        x = self.relu(x)
        x = self.l1(x)
        x = self.relu(x)
        out = self.l2(x)
        return out