import torch.nn as nn

class SIMPLEMODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(7, 128)
        self.layer2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.num_classes = 2

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
