from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RED_WINE_CSV = RAW_DATA_DIR / "winequality-red.csv"
WHITE_WINE_CSV = RAW_DATA_DIR / "winequality-white.csv"

QUALITY_THRESHOLD = 7
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25
