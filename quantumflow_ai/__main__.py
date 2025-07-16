import argparse
import sys
from pathlib import Path

# Allow execution both as ``python -m quantumflow_ai`` and
# ``python quantumflow_ai/__main__.py`` by ensuring the package root is on the
# path when run directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "quantumflow_ai"

# Support calling this CLI using the training script paths, e.g.
# ``python -m quantumflow_ai notebooks/train_meta_model.py``.
# Rewrite to corresponding subcommand.
alias_map = {
    "train_meta_model.py": "train-meta",
    "train_meta_model": "train-meta",
    "train_gnn_model.py": "train-gnn",
    "train_gnn_model": "train-gnn",
}
if len(sys.argv) > 1:
    first = Path(sys.argv[1]).name
    if first in alias_map:
        sys.argv[1] = alias_map[first]


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantumFlow utilities")
    subparsers = parser.add_subparsers(dest="command")

    # GNN Trainer
    gnn = subparsers.add_parser("train-gnn", help="Train the GNN energy model")
    gnn.add_argument("--profiles", nargs="+", default=["a100", "h100", "gb200"], help="Hardware profile names")
    gnn.add_argument("--data-dir", default="notebooks/profiles", help="Directory containing profile JSON files")
    gnn.add_argument("--model-out", default="modules/q_energy/model/gnn.pt", help="Path to save the trained model")

    # Meta Model Trainer
    meta = subparsers.add_parser("train-meta", help="Train the meta scheduler")
    meta.add_argument("--profiles", nargs="+", default=["a100", "h100", "gb200"], help="Hardware profile names")
    meta.add_argument("--data-dir", default="notebooks/profiles", help="Directory containing profile JSON files")

    args = parser.parse_args()

    if args.command == "train-gnn":
        from quantumflow_ai.notebooks.train_gnn_model import train_gnn
        train_gnn(args.profiles, args.data_dir, args.model_out)
    elif args.command == "train-meta":
        from quantumflow_ai.notebooks.train_meta_model import train_meta_from_profiles
        train_meta_from_profiles(args.profiles, args.data_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


   
