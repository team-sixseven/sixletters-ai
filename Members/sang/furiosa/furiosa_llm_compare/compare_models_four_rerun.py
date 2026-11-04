"""Compatibility entry point for the four-model comparison."""

import argparse
from pathlib import Path

from compare_models import MODELS, main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
