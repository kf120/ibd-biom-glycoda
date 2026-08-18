from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR.parent

DATASETS_DIR = BASE_DIR / "datasets"
FIGURE_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "outputs"