import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

device = "cpu"


# =========================================
# COLBERT-STYLE EMBEDDER
# =========================================

class ColBERTEmbedder:

    def __init__(self, model_name="bert-base-uncased"):

        print("Loading ColBERT-style model...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        self.model.eval()

    def encode(self, texts, batch_size=128):

        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size)):

            batch = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # CLS token embedding (fast approximation)
            emb = outputs.last_hidden_state[:, 0].cpu().numpy()

            embeddings.append(emb)

        return np.vstack(embeddings)


# =========================================
# CACHE EMBEDDINGS
# =========================================

def get_embeddings(embedder, texts, path):

    if os.path.exists(path):
        print(f"Loading cached embeddings: {path}")
        return np.load(path)

    print(f"Computing embeddings: {path}")
    emb = embedder.encode(texts)

    np.save(path, emb)
    return emb


# =========================================
# MAIN FUNCTION
# =========================================

def train_colbert_mixtral():

    print("="*60)
    print("FAST ColBERT + Retrieval Pipeline")
    print("="*60)

    train_df = pd.read_csv("data/splits/train.csv")
    val_df   = pd.read_csv("data/splits/val.csv")
    test_df  = pd.read_csv("data/splits/test.csv")

    embedder = ColBERTEmbedder()

    # =========================================
    # EMBEDDINGS (CACHED)
    # =========================================

    train_emb = get_embeddings(embedder, train_df["text"].astype(str).tolist(), "colbert_train.npy")
    val_emb   = get_embeddings(embedder, val_df["text"].astype(str).tolist(), "colbert_val.npy")
    test_emb  = get_embeddings(embedder, test_df["text"].astype(str).tolist(), "colbert_test.npy")

    # =========================================
    # CLASSIFIERS
    # =========================================

    print("Training classifiers...")

    fake_clf = LogisticRegression(max_iter=1000)
    hate_clf = LogisticRegression(max_iter=1000)

    fake_clf.fit(train_emb, train_df["Fake"])
    hate_clf.fit(train_emb, train_df["Hate"])

    # =========================================
    # TARGET / SEVERITY (FIX NaN)
    # =========================================

    hate_mask = train_df["Hate"] == 1

    target_data = train_df.loc[hate_mask].dropna(subset=["Target"])
    severity_data = train_df.loc[hate_mask].dropna(subset=["Severity"])

    target_emb = train_emb[target_data.index]
    severity_emb = train_emb[severity_data.index]

    target_clf = LogisticRegression(max_iter=1000)
    severity_clf = LogisticRegression(max_iter=1000)

    target_clf.fit(target_emb, target_data["Target"])
    severity_clf.fit(severity_emb, severity_data["Severity"])

    # =========================================
    # PREDICTIONS
    # =========================================

    print("Running predictions...")

    fake_pred = fake_clf.predict(test_emb)
    hate_pred = hate_clf.predict(test_emb)

    target_pred = []
    severity_pred = []

    for i in range(len(test_emb)):

        if hate_pred[i] == 1:

            t = target_clf.predict(test_emb[i].reshape(1,-1))[0]
            s = severity_clf.predict(test_emb[i].reshape(1,-1))[0]

        else:
            t = "None"
            s = "None"

        target_pred.append(t)
        severity_pred.append(s)

    # =========================================
    # METRICS
    # =========================================

    print("\nEvaluation")

    fake_f1 = f1_score(test_df["Fake"], fake_pred, average="macro")
    hate_f1 = f1_score(test_df["Hate"], hate_pred, average="macro")

    mask = (test_df["Hate"] == 1) & test_df["Target"].notna() & test_df["Severity"].notna()

    target_f1 = f1_score(
        test_df.loc[mask, "Target"],
        np.array(target_pred)[mask],
        average="macro"
    )

    severity_f1 = f1_score(
        test_df.loc[mask, "Severity"],
        np.array(severity_pred)[mask],
        average="macro"
    )

    print(f"Fake F1: {fake_f1:.4f}")
    print(f"Hate F1: {hate_f1:.4f}")
    print(f"Target F1: {target_f1:.4f}")
    print(f"Severity F1: {severity_f1:.4f}")

    # =========================================
    # SAVE RESULTS
    # =========================================

    os.makedirs("results/rag", exist_ok=True)

    results = {
        "Fake_F1": float(fake_f1),
        "Hate_F1": float(hate_f1),
        "Target_F1": float(target_f1),
        "Severity_F1": float(severity_f1)
    }

    with open("results/rag/colbert_mixtral_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved → results/rag/colbert_mixtral_results.json")
