import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=64):
        super(NeuralNet, self).__init__()
        # First hidden layer with dropout for regularization
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        # Optional second hidden layer (can be omitted or kept small)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        # Output layer
        self.l3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.l3(out)
        return out
