import torch.nn as nn


class DistilBERTClassifier(nn.Module):
    def __init__(self, embed_size, num_classes):
        super(DistilBERTClassifier, self).__init__()
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = x[:, 0, :]  # Take the [CLS] token
        logits = self.fc(x)
        return logits
