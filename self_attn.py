import torch.nn as nn
import torch


class SelfAttn(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttn, self).__init__()
        self.embed_size = embed_size
        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.embed_size, dtype=torch.float32)
        )
        attention = torch.softmax(scores, dim=-1)
        return self.fc(torch.matmul(attention, V))
