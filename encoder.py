import torch.nn as nn
from self_attn import SelfAttn
from feed_forward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(EncoderLayer, self).__init__()
        self.self_attn = SelfAttn(embed_size)
        self.feed_forward = FeedForward(embed_size, hidden_size)
        self.norm1 = nn.Linear(embed_size)
        self.norm2 = nn.Linear(embed_size)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x
