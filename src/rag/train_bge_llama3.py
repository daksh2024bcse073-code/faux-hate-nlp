import os
import json
import torch
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


device = "cpu"


# =========================================
# BGE EMBEDDINGS
# =========================================

class BGEEmbedder:

    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):

        print("Loading BGE embedding model...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        self.model.eval()

    def encode(self, texts, batch_size=32):

        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size)):

            batch = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=180,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            emb = outputs.last_hidden_state[:,0].cpu().numpy()

            embeddings.append(emb)

        embeddings = np.vstack(embeddings)

        return embeddings


# =========================================
# MAIN FUNCTION
# =========================================

def train_bge_llama3():

    print("="*60)
    print("FAST BGE Retrieval Pipeline")
    print("="*60)

    train_df = pd.read_csv("data/splits/train.csv")
    val_df   = pd.read_csv("data/splits/val.csv")
    test_df  = pd.read_csv("data/splits/test.csv")

    embedder = BGEEmbedder()

    print("Encoding train set...")
    train_embeddings = embedder.encode(train_df["text"].astype(str).tolist())

    print("Encoding val set...")
    val_embeddings = embedder.encode(val_df["text"].astype(str).tolist())

    print("Encoding test set...")
    test_embeddings = embedder.encode(test_df["text"].astype(str).tolist())

    # =========================================
    # FAISS INDEX
    # =========================================

    dim = train_embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(train_embeddings)

    print("FAISS index built.")

    # =========================================
    # CLASSIFIERS
    # =========================================

    print("Training classifiers...")

    fake_clf = LogisticRegression(max_iter=1000)
    hate_clf = LogisticRegression(max_iter=1000)

    fake_clf.fit(train_embeddings, train_df["Fake"])
    hate_clf.fit(train_embeddings, train_df["Hate"])

    # Train Target / Severity only on hate samples
    hate_mask = train_df["Hate"] == 1

    target_clf = LogisticRegression(max_iter=1000)
    severity_clf = LogisticRegression(max_iter=1000)

    target_clf.fit(
        train_embeddings[hate_mask],
        train_df.loc[hate_mask, "Target"]
    )

    severity_clf.fit(
        train_embeddings[hate_mask],
        train_df.loc[hate_mask, "Severity"]
    )

    # =========================================
    # PREDICTIONS
    # =========================================

    print("Running predictions...")

    fake_pred = fake_clf.predict(test_embeddings)
    hate_pred = hate_clf.predict(test_embeddings)

    target_pred = []
    severity_pred = []

    for i in range(len(test_embeddings)):

        if hate_pred[i] == 1:

            t = target_clf.predict(test_embeddings[i].reshape(1,-1))[0]
            s = severity_clf.predict(test_embeddings[i].reshape(1,-1))[0]

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

    print("Fake Macro F1:", fake_f1)
    print("Hate Macro F1:", hate_f1)

    mask = test_df["Hate"] == 1

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

    print("Target Macro F1:", target_f1)
    print("Severity Macro F1:", severity_f1)

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

    with open("results/rag/bge_llama3_results.json","w") as f:
        json.dump(results,f,indent=4)

    print("\nResults saved to results/rag/bge_llama3_results.json")
