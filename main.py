import argparse
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from train_roberta import train_roberta
from train_xlmroberta import train_xlmroberta
from train_deberta import train_deberta
from src.rag.train_bge_llama3 import train_bge_llama3
from src.rag.train_colbert_mixtral import train_colbert_mixtral

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=["roberta", "xlmroberta", "deberta", "bge_llama3"],
        required=True
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()

    if args.model == "roberta":
        train_roberta(args.epochs, args.batch_size, args.lr)

    elif args.model == "xlmroberta":
        train_xlmroberta(args.epochs, args.batch_size, args.lr)

    elif args.model == "deberta":
        train_deberta(args.epochs, args.batch_size, args.lr)

    elif args.model == "bge_llama3":
        train_bge_llama3()


if __name__ == "__main__":
    main()
