import torch.nn as nn
from embeddings import Embeddings
from distil_encoder import DistilBERTEncoder
from classifier import DistilBERTClassifier


class DistilBERT(nn.Module):
    def __init__(
        self, vocab_size, embed_size, hidden_size, num_layers, num_classes, max_length
    ):
        super(DistilBERT, self).__init__()
        self.embedding = Embeddings(vocab_size, embed_size, max_length)
        self.encoder = DistilBERTEncoder(embed_size, hidden_size, num_layers)
        self.classifier = DistilBERTClassifier(embed_size, num_classes)

    def forward(self, x, segment_ids, mask=None):
        embeddings = self.embedding(x, segment_ids)
        encoder_output = self.encoder(embeddings, mask)
        logits = self.classifier(encoder_output)
        return logits
