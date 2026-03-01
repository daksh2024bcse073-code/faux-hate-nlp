import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    DebertaV2Model,
    DebertaV2TokenizerFast,
    get_linear_schedule_with_warmup,
)

from data_utils import FauxHateDataset, IGNORE_INDEX


# =========================================
# Multi-Task DeBERTa Model
# =========================================
class MultiTaskDeberta(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()

      self.encoder = DebertaV2Model.from_pretrained(
    "microsoft/deberta-v3-base",
    torch_dtype=torch.float32
)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.3)

        # Task heads
        self.fake_head = nn.Linear(hidden_size, 2)
        self.hate_head = nn.Linear(hidden_size, 2)
        self.target_head = nn.Linear(hidden_size, 3)
        self.severity_head = nn.Linear(hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)

        fake_logits = self.fake_head(cls_output)
        hate_logits = self.hate_head(cls_output)
        target_logits = self.target_head(cls_output)
        severity_logits = self.severity_head(cls_output)

        return fake_logits, hate_logits, target_logits, severity_logits


# =========================================
# Training Function
# =========================================
def train_deberta(
    epochs=3,
    batch_size=16,
    lr=2e-5,
    device="cpu"
):
    print("=" * 60)
    print("Training DeBERTa-v3")
    print("Device:", device)
    print("=" * 60)

    tokenizer = DebertaV2TokenizerFast.from_pretrained(
        "microsoft/deberta-v3-base"
    )

    train_dataset = FauxHateDataset(
        csv_path="data/splits/train.csv",
        tokenizer=tokenizer,
        max_len=180
    )

    val_dataset = FauxHateDataset(
        csv_path="data/splits/val.csv",
        tokenizer=tokenizer,
        max_len=180
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = MultiTaskDeberta().to(device)
    model = model.float()

    optimizer = AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # =========================================
    # Training Loop
    # =========================================
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            fake = batch["fake"].to(device)
            hate = batch["hate"].to(device)
            target = batch["target"].to(device)
            severity = batch["severity"].to(device)

            optimizer.zero_grad()

            fake_logits, hate_logits, target_logits, severity_logits = model(
                input_ids,
                attention_mask
            )

            loss_fake = criterion(fake_logits, fake)
            loss_hate = criterion(hate_logits, hate)
            loss_target = criterion(target_logits, target)
            loss_severity = criterion(severity_logits, severity)

            loss = loss_fake + loss_hate + loss_target + loss_severity

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # =========================================
    # Save Model
    # =========================================
    os.makedirs("outputs/deberta", exist_ok=True)
    torch.save(model.state_dict(), "outputs/deberta/model.pt")

    os.makedirs("results/metrics", exist_ok=True)

    metrics = {
        "model": "DeBERTa-v3-base",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "note": "Training completed successfully"
    }

    with open("results/metrics/deberta_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Training complete.")
    print("Model saved to outputs/deberta/")
    print("Metrics saved to results/metrics/deberta_metrics.json")


# =========================================
# Run Directly
# =========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_deberta(device=device)

