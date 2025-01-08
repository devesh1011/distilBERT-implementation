import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from model import DistilBERT
from utils.functions import tokenize_function, distillation_loss

dataset = load_dataset("glue", "sst2")
teacher_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
)

train_loader = DataLoader(tokenized_dataset["train"], batch_size=16, shuffle=True)

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


ce_loss = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
teacher_model.to(device)

for epoch in range(num_epochs):
    model.train()  
    total_loss = 0  
    total_correct = 0  
    total_samples = 0  

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            teacher_outputs = teacher_model(**batch)
            teacher_logits = teacher_outputs.logits

        student_logits = model(
            batch["input_ids"], batch["token_type_ids"], mask=batch["attention_mask"]
        )

        loss_ce = ce_loss(student_logits, batch["labels"])
        loss_distill = distillation_loss(student_logits, teacher_logits, temperature=2)
        loss = loss_ce + loss_distill

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(student_logits, dim=1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
