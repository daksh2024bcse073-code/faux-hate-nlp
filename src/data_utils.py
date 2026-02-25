import pandas as pd
import torch
from torch.utils.data import Dataset

# Label mappings
TARGET_MAP = {"I": 0, "O": 1, "R": 2}
SEVERITY_MAP = {"L": 0, "M": 1, "H": 2}

IGNORE_INDEX = -100  # for missing Target/Severity


class FauxHateDataset(Dataset):
    """
    Expects CSV with columns:
    id, text, Fake, Hate, Target, Severity
    """
    def __init__(self, csv_path, tokenizer, max_len=180):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Basic checks
        required_cols = ["text", "Fake", "Hate", "Target", "Severity"]
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing column '{c}' in {csv_path}")

        # Fill NaN with empty string for safe processing
        self.df["Target"] = self.df["Target"].fillna("")
        self.df["Severity"] = self.df["Severity"].fillna("")

    def __len__(self):
        return len(self.df)

    def _encode_target(self, val):
        if val == "" or val not in TARGET_MAP:
            return IGNORE_INDEX
        return TARGET_MAP[val]

    def _encode_severity(self, val):
        if val == "" or val not in SEVERITY_MAP:
            return IGNORE_INDEX
        return SEVERITY_MAP[val]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row["text"])
        fake = int(row["Fake"])
        hate = int(row["Hate"])

        # Encode Target/Severity (ignore if empty)
        target = self._encode_target(str(row["Target"]).strip())
        severity = self._encode_severity(str(row["Severity"]).strip())

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