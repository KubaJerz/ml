import torch
import torch.nn as nn

class MODEL(nn.Module):
    def __init__(self, input_channels, output_size, hyperparams):
        super(MODEL, self).__init__()
        self.hidden_blocks = hyperparams['hidden_blocks']
        self.layer_depth = hyperparams['layer_depth']
        self.dropout_rate = hyperparams['dropout_rate']
        self.activation = getattr(nn, hyperparams['activation'])()
        self.normalization = hyperparams['normalization']

        layers = []
        in_channels = input_channels

        for block, block_layers in enumerate(self.layer_depth):
            for layer, out_channels in enumerate(block_layers):
                stride = 2 if layer == 0 else 1
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
                
                if self.normalization == 'batch':
                    layers.append(nn.BatchNorm2d(out_channels))
                elif self.normalization == 'layer':
                    layers.append(nn.LayerNorm(1, out_channels))
                
                layers.append(self.activation)
                
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout2d(self.dropout_rate))
                
                in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)


        self.fc1 = nn.Linear(in_channels * (32 // (2 ** len(self.layer_depth))) ** 2, 1000)
        self.fc2 = nn.Linear(1000, output_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
