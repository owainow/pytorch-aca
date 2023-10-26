import torch.nn as nn
import torch

# Define neural network
class Net (nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 28*28) #Flattens input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x