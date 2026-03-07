import os
import json
import torch
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================
# BGE EMBEDDING MODEL
# ================================

class BGEEmbedder:

    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def encode(self, texts):

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0]
        return embeddings.cpu().numpy()


# ================================
# LOAD LLM (MISTRAL / LLAMA STYLE)
# ================================

def load_llm():

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    return tokenizer, model


# ================================
# PROMPT CLASSIFICATION
# ================================

def classify_text(tokenizer, model, text):

    prompt = f"""
You are a classifier.

Classify the following text into:

Fake: 0 or 1
Hate: 0 or 1
Target: I (Individual), O (Community), R (Religion)
Severity: L (Low), M (Medium), H (High)

Text:
{text}

Return answer in JSON format.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.0
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result


# ================================
# TRAIN / EVALUATE RAG
# ================================

def train_bge_llama3():

    print("="*60)
    print("Running BGE + LLM RAG pipeline")
    print("Device:", device)
    print("="*60)

    df = pd.read_csv("data/splits/test.csv")

    texts = df["text"].astype(str).tolist()

    # Load embedding model
    embedder = BGEEmbedder()

    print("Generating embeddings...")

    embeddings = embedder.encode(texts)

    # Build FAISS index
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("FAISS index built")

    # Load LLM
    tokenizer, model = load_llm()

    predictions = []

    for text in tqdm(texts[:200]):   # limit for speed

        result = classify_text(tokenizer, model, text)

        predictions.append({
            "text": text,
            "prediction": result
        })

    os.makedirs("results/rag", exist_ok=True)

    with open("results/rag/bge_llm_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print("Results saved to results/rag/bge_llm_predictions.json")
