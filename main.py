# embeddings
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from model import DistilBERT

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


vocab_size = tokenizer.vocab_size
embed_size = 768  # Example embedding size
hidden_size = 4 * embed_size
max_length = 128
num_layers = 6
num_classes = 2
model = DistilBERT(
    vocab_size, embed_size, hidden_size, num_layers, num_classes, max_length
)
num_epochs = 1
teacher_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

ce_loss = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
teacher_model.to(device)

train_loader = DataLoader(tokenized_dataset["train"], batch_size=16, shuffle=True)

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0  # Track total loss for the epoch
    total_correct = 0  # Track total correct predictions
    total_samples = 0  # Track total samples processed

    for batch in train_loader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Teacher model outputs (no gradient needed)
        with torch.no_grad():
            teacher_outputs = teacher_model(**batch)
            teacher_logits = teacher_outputs.logits

        # Student model outputs
        student_logits = model(
            batch["input_ids"], batch["token_type_ids"], mask=batch["attention_mask"]
        )

        # Compute losses
        loss_ce = ce_loss(student_logits, batch["labels"])
        loss_distill = distillation_loss(student_logits, teacher_logits, temperature=2)
        loss = loss_ce + loss_distill

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        total_loss += loss.item()
        preds = torch.argmax(student_logits, dim=1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    # Print results for the epoch
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
