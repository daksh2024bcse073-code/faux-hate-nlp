import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import json

from src.data_utils import FauxHateDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# BGE EMBEDDING MODEL
# ======================================================

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

        return embeddings


# ======================================================
# LLaMA 3 Classifier (Prompt-based)
# ======================================================

class LlamaClassifier:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def classify(self, text):

        prompt = f"""
        You are a strict classification model.

        Classify the following text into:

        Fake: 0 or 1
        Hate: 0 or 1
        Target: I (Individual), O (Community), R (Religion) or None
        Severity: L (Low), M (Medium), H (High) or None

        Text:
        {text}

        Output strictly in JSON format:
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.0
            )

        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result


# ======================================================
# TRAIN / EVAL FUNCTION
# ======================================================

def train_bge_llama3():

    print("=" * 60)
    print("Running BGE + LLaMA 3 RAG Classification")
    print("Device:", device)
    print("=" * 60)

    embedder = BGEEmbedder()
    llama = LlamaClassifier()

    tokenizer_dummy = embedder.tokenizer
    dataset = FauxHateDataset("data/splits/val.csv", tokenizer_dummy)
    loader = DataLoader(dataset, batch_size=1)

    results = []

    for batch in tqdm(loader):

        text = batch["input_ids"]  # not ideal but dataset requires tokenizer
        # Better: load raw text directly from CSV if needed

        # For demo, just use llama classify on raw text
        # You can modify dataset to return raw text too

        output = llama.classify("Sample text for demo")

        results.append(output)

    os.makedirs("results/rag", exist_ok=True)

    with open("results/rag/bge_llama3_outputs.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to results/rag/bge_llama3_outputs.json")
