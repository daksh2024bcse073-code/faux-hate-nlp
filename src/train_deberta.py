import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DebertaV2Model, DebertaV2Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from src.data_utils import FauxHateDataset, IGNORE_INDEX


# ====================================================
# Multi-task Model
# ====================================================

class DebertaMultiTask(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()

        self.encoder = DebertaV2Model.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

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

        fake_logits = self.fake_head(cls)
        hate_logits = self.hate_head(cls)
        target_logits = self.target_head(cls)
        severity_logits = self.severity_head(cls)

        return fake_logits, hate_logits, target_logits, severity_logits


# ====================================================
# Training Function
# ====================================================

def train_deberta(epochs, batch_size, lr, device):

    print("=" * 60)
    print("Training DeBERTa-v3")
    print("Device:", device)
    print("=" * 60)

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")

    train_dataset = FauxHateDataset("data/splits/train.csv", tokenizer)
    val_dataset = FauxHateDataset("data/splits/val.csv", tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = DebertaMultiTask().to(device)
    model = model.float()

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

    # ===============================
    # TRAIN LOOP
    # ===============================

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

            fake_logits, hate_logits, target_logits, severity_logits = model(
                input_ids, attention_mask
            )

            loss_fake = fake_loss_fn(fake_logits, fake_labels)
            loss_hate = hate_loss_fn(hate_logits, hate_labels)
            loss_target = target_loss_fn(target_logits, target_labels)
            loss_severity = severity_loss_fn(severity_logits, severity_labels)

            loss = loss_fake + loss_hate + loss_target + loss_severity

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

    # ===============================
    # SAVE MODEL
    # ===============================

    os.makedirs("outputs/deberta", exist_ok=True)
    torch.save(model.state_dict(), "outputs/deberta/model.pt")

    print("Model saved to outputs/deberta")

    # ===============================
    # SAVE SIMPLE METRICS
    # ===============================

    os.makedirs("results/metrics", exist_ok=True)

    metrics = {
        "training_complete": True
    }

    with open("results/metrics/deberta_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics saved to results/metrics/deberta_metrics.json")
