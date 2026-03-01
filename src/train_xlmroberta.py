import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from src.data_utils import FauxHateDataset, IGNORE_INDEX


# ====================================================
# Multi-task Model
# ====================================================

class XLMRMultiTask(nn.Module):
    def __init__(self, model_name="xlm-roberta-base"):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        self.fake_head = nn.Linear(hidden_size, 2)
        self.hate_head = nn.Linear(hidden_size, 2)
        self.target_head = nn.Linear(hidden_size, 3)
        self.severity_head = nn.Linear(hidden_size, 3)

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0]

        return {
            "fake": self.fake_head(cls),
            "hate": self.hate_head(cls),
            "target": self.target_head(cls),
            "severity": self.severity_head(cls),
        }


# ====================================================
# Training Function (MATCHES main.py CALL)
# ====================================================

def train_xlmroberta(epochs, batch_size, lr, device):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("Using device:", device)
    print("=" * 50)

    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

    train_dataset = FauxHateDataset("data/splits/train.csv", tokenizer)
    val_dataset = FauxHateDataset("data/splits/val.csv", tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = XLMRMultiTask().to(device)

    # Loss functions
    fake_loss_fn = nn.CrossEntropyLoss()
    hate_loss_fn = nn.CrossEntropyLoss()
    target_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    severity_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    optimizer = AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # ====================================================
    # TRAIN LOOP
    # ====================================================

    for epoch in range(epochs):

        model.train()
        epoch_loss = 0.0
        valid_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            fake_labels = batch["fake"].to(device)
            hate_labels = batch["hate"].to(device)
            target_labels = batch["target"].to(device)
            severity_labels = batch["severity"].to(device)

            outputs = model(input_ids, attention_mask)

            loss_fake = fake_loss_fn(outputs["fake"], fake_labels)
            loss_hate = hate_loss_fn(outputs["hate"], hate_labels)
            loss_target = target_loss_fn(outputs["target"], target_labels)
            loss_severity = severity_loss_fn(outputs["severity"], severity_labels)

            loss = loss_fake + loss_hate + loss_target + loss_severity

            # Safety check
            if torch.isnan(loss):
                print("NaN detected — skipping batch")
                continue

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            valid_batches += 1

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0.0
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    print("Training complete.")

    # ====================================================
    # SAVE MODEL
    # ====================================================

    os.makedirs("outputs/xlmroberta", exist_ok=True)
    torch.save(model.state_dict(), "outputs/xlmroberta/model.pt")

    print("Model saved to outputs/xlmroberta")

    # ====================================================
    # SIMPLE VALIDATION (Fake task only for now)
    # ====================================================

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            fake_labels = batch["fake"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs["fake"], dim=1)

            correct += (preds == fake_labels).sum().item()
            total += fake_labels.size(0)

    accuracy = correct / total

    os.makedirs("results/metrics", exist_ok=True)

    metrics = {
        "validation_accuracy_fake": accuracy
    }

    with open("results/metrics/xlmroberta_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics saved to results/metrics/xlmroberta_metrics.json")
    print("Validation Fake Accuracy:", accuracy)

