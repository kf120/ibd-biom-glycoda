# config.py
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# One level up from the project directory
BASE_DIR = PROJECT_DIR.parent

# Shared directories at repository root
DATASETS_DIR = BASE_DIR / "datasets"
FIGURE_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "outputs"