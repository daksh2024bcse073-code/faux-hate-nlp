import argparse
import torch
import os

from src.train_roberta import train_roberta
# Later we will add:
# from src.train_xlmroberta import train_xlmroberta
# from src.train_deberta import train_deberta


def main():
    parser = argparse.ArgumentParser(description="Faux Hate NLP Training")

    parser.add_argument("--model", type=str, required=True, choices=["roberta", "xlmroberta", "deberta"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=180)

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("====================================")
    print("Using device:", device)
    print("====================================")

    # Data paths
    train_file = os.path.join("data", "splits", "train.csv")
    val_file = os.path.join("data", "splits", "val.csv")
    test_file = os.path.join("data", "splits", "test.csv")

    output_dir = "outputs"

    if args.model == "roberta":
        train_roberta(
            train_file=train_file,
            val_file=val_file,
            test_file=test_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_len=args.max_len,
            output_dir=output_dir,
            device=device,
        )
    else:
        raise NotImplementedError(f"Model {args.model} not implemented yet!")


if __name__ == "__main__":
    main()
