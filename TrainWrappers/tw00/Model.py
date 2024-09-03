import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, output_size, hyperparams):
        super(Model, self).__init__()
        
        self.hidden_blocks = hyperparams['hidden_blocks']
        self.layer_depth = hyperparams['neurons_per_layer']
        self.dropout_rate = hyperparams['dropout_rate']
        self.activation = getattr(nn, hyperparams['activation'])()
        self.normalization = hyperparams['normalization']

        
        # Input layer
        layers = [nn.Linear(input_size, self.neurons_per_layer)]

        for block in range(self.hidden_blocks):
            
        
        if self.normalization == 'batch':
            layers.append(nn.BatchNorm1d(self.neurons_per_layer))
        
        layers.append(self.activation)
        layers.append(nn.Dropout(self.dropout_rate))
        
        # Hidden layers
        for _ in range(self.hidden_blocks - 1):
            layers.append(nn.Linear(self.neurons_per_layer, self.neurons_per_layer))
            
            if self.normalization == 'batch':
                layers.append(nn.BatchNorm1d(self.neurons_per_layer))
            
            layers.append(self.activation)
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(self.neurons_per_layer, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Example usage:
input_size = 10  # Replace with your input size
output_size = 1  # Replace with your output size

model = CustomModel(input_size, output_size, hyperparams)