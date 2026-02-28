import argparse
import torch
import sys
import os

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# from train_roberta import train_roberta
from train_xlmroberta import train_xlmroberta
# from train_deberta import train_deberta


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True,
                        choices=["roberta", "xlmroberta", "deberta"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 40)
    print("Using device:", device)
    print("=" * 40)

    if args.model == "roberta":
        train_roberta(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device
        )

    elif args.model == "xlmroberta":
        train_xlmroberta(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device
        )

    elif args.model == "deberta":
        train_deberta(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device
        )

    else:
        raise NotImplementedError(f"Model {args.model} not implemented yet!")


if __name__ == "__main__":
    main()






