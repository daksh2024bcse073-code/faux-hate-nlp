import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from src.data_utils import FauxHateDataset, IGNORE_INDEX


class MultiHeadRoberta(nn.Module):
    """
    4 heads:
    - Fake (2 classes)
    - Hate (2 classes)
    - Target (3 classes: I/O/R)
    - Severity (3 classes: L/M/H)
    """
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.backbone = RobertaModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        self.fake_head = nn.Linear(hidden, 2)
        self.hate_head = nn.Linear(hidden, 2)
        self.target_head = nn.Linear(hidden, 3)
        self.severity_head = nn.Linear(hidden, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)

        fake_logits = self.fake_head(pooled)
        hate_logits = self.hate_head(pooled)
        target_logits = self.target_head(pooled)
        severity_logits = self.severity_head(pooled)

        return fake_logits, hate_logits, target_logits, severity_logits


def compute_metrics(y_true, y_pred, average="macro"):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average=average),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    fake_true, fake_pred = [], []
    hate_true, hate_pred = [], []
    target_true, target_pred = [], []
    severity_true, severity_pred = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        fake = batch["fake"].to(device)
        hate = batch["hate"].to(device)
        target = batch["target"].to(device)
        severity = batch["severity"].to(device)

        fake_logits, hate_logits, target_logits, severity_logits = model(input_ids, attention_mask)

        fake_p = torch.argmax(fake_logits, dim=1)
        hate_p = torch.argmax(hate_logits, dim=1)
        target_p = torch.argmax(target_logits, dim=1)
        severity_p = torch.argmax(severity_logits, dim=1)

        fake_true.extend(fake.cpu().numpy().tolist())
        fake_pred.extend(fake_p.cpu().numpy().tolist())

        hate_true.extend(hate.cpu().numpy().tolist())
        hate_pred.extend(hate_p.cpu().numpy().tolist())

        # Only evaluate Target/Severity where label != IGNORE_INDEX
        for t_true, t_pred in zip(target, target_p):
            if t_true.item() != IGNORE_INDEX:
                target_true.append(t_true.item())
                target_pred.append(t_pred.item())

        for s_true, s_pred in zip(severity, severity_p):
            if s_true.item() != IGNORE_INDEX:
                severity_true.append(s_true.item())
                severity_pred.append(s_pred.item())

    metrics = {}

    metrics["fake"] = compute_metrics(fake_true, fake_pred)
    metrics["hate"] = compute_metrics(hate_true, hate_pred)

    if len(target_true) > 0:
        metrics["target"] = compute_metrics(target_true, target_pred)
    else:
        metrics["target"] = {"accuracy": None, "macro_f1": None}

    if len(severity_true) > 0:
        metrics["severity"] = compute_metrics(severity_true, severity_pred)
    else:
        metrics["severity"] = {"accuracy": None, "macro_f1": None}

    return metrics


def train_roberta(
    train_file,
    val_file,
    test_file,
    epochs,
    batch_size,
    lr,
    max_len,
    output_dir,
    device
):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    print("Loading datasets...")
    train_ds = FauxHateDataset(train_file, tokenizer, max_len=max_len)
    val_ds = FauxHateDataset(val_file, tokenizer, max_len=max_len)
    test_ds = FauxHateDataset(test_file, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Building model...")
    model = MultiHeadRoberta().to(device)

    # Losses
    ce_fake = nn.CrossEntropyLoss()
    ce_hate = nn.CrossEntropyLoss()
    ce_target = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    ce_severity = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    optimizer = AdamW(model.parameters(), lr=lr)

    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    print("====================================")
    print("Training RoBERTa (multi-task)")
    print("Device:", device)
    print("Epochs:", epochs, "Batch size:", batch_size, "LR:", lr)
    print("====================================")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            fake = batch["fake"].to(device)
            hate = batch["hate"].to(device)
            target = batch["target"].to(device)
            severity = batch["severity"].to(device)

            fake_logits, hate_logits, target_logits, severity_logits = model(input_ids, attention_mask)

            loss_fake = ce_fake(fake_logits, fake)
            loss_hate = ce_hate(hate_logits, hate)
            loss_target = ce_target(target_logits, target)
            loss_severity = ce_severity(severity_logits, severity)

            loss = loss_fake + loss_hate + loss_target + loss_severity
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")

        print("Validating...")
        val_metrics = evaluate(model, val_loader, device)
        print("Validation metrics:", val_metrics)

    print("\nTesting on test set...")
    test_metrics = evaluate(model, test_loader, device)
    print("Test metrics:", test_metrics)

    # Save metrics
    metrics_path = os.path.join("results", "metrics", "roberta_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("Saved metrics to:", metrics_path)

    # Save model
    save_dir = os.path.join(output_dir, "roberta")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    print("Saved model to:", save_dir)