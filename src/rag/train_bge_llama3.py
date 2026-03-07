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


# ===============================
# BGE EMBEDDING MODEL (CPU)
# ===============================

class BGEEmbedder:

    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        print("Loading BGE embedding model on CPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to("cpu")
        self.model.eval()

    def encode(self, texts, batch_size=32):

        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=180,
                return_tensors="pt"
            ).to("cpu")

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_embeddings = outputs.last_hidden_state[:,0].cpu().numpy()
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)

        return embeddings


# ===============================
# LOAD MISTRAL (LLM)
# ===============================

def load_llm():

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    print("Loading Mistral LLM...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    return tokenizer, model


# ===============================
# LLM CLASSIFICATION
# ===============================

def classify_text(tokenizer, model, text, context):

    prompt = f"""
You are a text classifier.

Use the retrieved context to help classify the text.

Context:
{context}

Text:
{text}

Return JSON format:

{{
"Fake": 0 or 1,
"Hate": 0 or 1,
"Target": "I" or "O" or "R",
"Severity": "L" or "M" or "H"
}}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.0
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    return result


# ===============================
# MAIN RAG PIPELINE
# ===============================

def train_bge_llama3():

    print("="*60)
    print("Running BGE + Mistral RAG pipeline")
    print("Device:", device)
    print("="*60)

    df = pd.read_csv("data/splits/test.csv")

    texts = df["text"].astype(str).tolist()

    # limit for demo
    texts = texts[:2000]

    # ===============================
    # EMBEDDINGS
    # ===============================

    embedder = BGEEmbedder()

    print("Generating embeddings...")

    embeddings = embedder.encode(texts)

    # ===============================
    # BUILD FAISS INDEX
    # ===============================

    dim = embeddings.shape[1]

    print("Building FAISS index...")

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("FAISS index ready")

    # ===============================
    # LOAD LLM
    # ===============================

    tokenizer, model = load_llm()

    # ===============================
    # RETRIEVAL + CLASSIFICATION
    # ===============================

    predictions = []

    for i, text in enumerate(tqdm(texts, desc="RAG inference")):

        query_embedding = embedder.encode([text])

        D, I = index.search(query_embedding, k=3)

        retrieved_context = "\n".join([texts[idx] for idx in I[0]])

        result = classify_text(
            tokenizer,
            model,
            text,
            retrieved_context
        )

        predictions.append({
            "text": text,
            "prediction": result
        })

    # ===============================
    # SAVE RESULTS
    # ===============================

    os.makedirs("results/rag", exist_ok=True)

    output_path = "results/rag/bge_mistral_predictions.json"

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print("Results saved to:", output_path)
