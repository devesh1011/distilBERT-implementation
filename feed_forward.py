import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedd_size, hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embedd_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embedd_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
