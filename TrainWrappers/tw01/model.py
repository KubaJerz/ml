import torch
import torch.nn as nn

class MODEL(nn.Module):
    def __init__(self, input_channels, output_size, hyperparams, num_classes):
        super(MODEL, self).__init__()
        self.hidden_blocks = hyperparams['hidden_blocks']
        self.layer_depth = hyperparams['layer_depth']
        self.dropout_rate = hyperparams['dropout_rate']
        self.activation = getattr(nn, hyperparams['activation'])()
        self.normalization = hyperparams['normalization']
        self.num_classes = num_classes

        layers = []
        in_channels = input_channels
        seq_length = 1000
        

        for block in range(self.hidden_blocks):
            for block_layer, out_channels in enumerate(self.layer_depth):
                stride = 2 if block_layer == 0 else 1
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
                seq_length = (seq_length + 2 * 1 - (3 - 1) - 1) // stride + 1

                if self.normalization == 'batch':
                    layers.append(nn.BatchNorm1d(out_channels))
                elif self.normalization == 'layer':
                    layers.append(nn.LayerNorm([out_channels, seq_length]))

                layers.append(self.activation)
                
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout1d(self.dropout_rate))
                
                in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)


        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 1000)
            dummy_output = self.feature_extractor(dummy_input)
            flattened_size = dummy_output.numel()
        
        self.fc1 = nn.Linear(flattened_size, 1000)
        self.fc2 = nn.Linear(1000, output_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
