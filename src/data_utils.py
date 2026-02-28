import pandas as pd
import torch
from torch.utils.data import Dataset

# =========================================
# Label mappings
# =========================================

IGNORE_INDEX = -100  # for ignored labels

TARGET_MAP = {"I": 0, "O": 1, "R": 2}
SEVERITY_MAP = {"L": 0, "M": 1, "H": 2}


# =========================================
# Dataset
# =========================================

class FauxHateDataset(Dataset):
    """
    Expects CSV with columns:
    id, text, Fake, Hate, Target, Severity
    """

    def __init__(self, csv_path, tokenizer, max_len=180):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        required_cols = ["text", "Fake", "Hate", "Target", "Severity"]
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing column '{c}' in {csv_path}")

        # Fill missing target/severity safely
        self.df["Target"] = self.df["Target"].fillna("")
        self.df["Severity"] = self.df["Severity"].fillna("")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        text = str(row["text"])
        fake = int(row["Fake"])
        hate = int(row["Hate"])

        # Only compute target/severity if Hate == 1
        if hate == 1:
            target = TARGET_MAP.get(str(row["Target"]).strip(), IGNORE_INDEX)
            severity = SEVERITY_MAP.get(str(row["Severity"]).strip(), IGNORE_INDEX)
        else:
            target = IGNORE_INDEX
            severity = IGNORE_INDEX

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "fake": torch.tensor(fake, dtype=torch.long),
            "hate": torch.tensor(hate, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "severity": torch.tensor(severity, dtype=torch.long),
        }

        return item
