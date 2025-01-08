import torch.nn as nn
from embeddings import Embeddings
from distil_encoder import DistilBERTEncoder
import torch
from classifier import DistilBERTClassifier


class DistilBERT(nn.Module):
    def __init__(
        self, vocab_size, embed_size, hidden_size, num_layers, num_classes, max_length
    ):
        super(DistilBERT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.encoder = DistilBERTEncoder(embed_size, hidden_size, num_layers)
        self.classifier = nn.Linear(embed_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        embeddings = self.embedding(input_ids) + self.position_embedding(position_ids)
        encoder_output = self.encoder(embeddings, mask=attention_mask)
        logits = self.classifier(encoder_output[:, 0, :])
        return logits
