import torch.nn.functional as F

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["sentence"], padding="max_length", truncation=True, max_length=128
    )


def distillation_loss(student_logits, teacher_logits, temperature):
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    student_logits_divT = student_logits / temperature
    return F.kl_div(
        F.log_softmax(student_logits_divT, dim=-1), soft_targets, reduction="batchmean"
    ) * (temperature**2)
