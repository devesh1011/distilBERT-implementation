import torch.nn as nn
from encoder import EncoderLayer


class DistilBERTEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        super(DistilBERTEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_size, hidden_size) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
