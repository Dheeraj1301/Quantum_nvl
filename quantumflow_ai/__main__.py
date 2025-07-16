import argparse
from quantumflow_ai.notebooks.train_gnn_model import train_gnn
from quantumflow_ai.notebooks.train_meta_model import train_meta_from_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantumFlow utilities")
    subparsers = parser.add_subparsers(dest="command")

    gnn = subparsers.add_parser("train-gnn", help="Train the GNN energy model")
    gnn.add_argument("--profiles", nargs="+", default=["a100", "h100", "gb200"], help="Hardware profile names")
    gnn.add_argument("--data-dir", default="notebooks/profiles", help="Directory containing profile JSON files")
    gnn.add_argument("--model-out", default="modules/q_energy/model/gnn.pt", help="Path to save the trained model")

    meta = subparsers.add_parser("train-meta", help="Train the meta scheduler")
    meta.add_argument("--profiles", nargs="+", default=["a100", "h100", "gb200"], help="Hardware profile names")
    meta.add_argument("--data-dir", default="notebooks/profiles", help="Directory containing profile JSON files")

    args = parser.parse_args()

    if args.command == "train-gnn":
        train_gnn(args.profiles, args.data_dir, args.model_out)
    elif args.command == "train-meta":
        train_meta_from_profiles(args.profiles, args.data_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
