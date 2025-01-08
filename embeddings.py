import torch.nn as nn
import torch

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)
        self.segment_embedding = nn.Embedding(2, embed_size)

    def forward(self, x, segment_ids):
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = self.token_embedding.unsqueeze(0).expand_as(x)
        embeddings = (
            self.token_embedding(x)
            + self.positional_embedding(position_ids)
            + self.segment_embedding(segment_ids)
        )
        return embeddings
