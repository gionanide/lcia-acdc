"""Fixed scientific settings for the historical four-model analysis.

Input and output locations are supplied by the analysis entry points.
"""
from pathlib import Path

# Paths are replaced by command-line arguments before any data is opened.
DATA_ROOT = Path("ACDC")
TRAIN_DIR = DATA_ROOT / "training"     # patients 001-100 -> train + val
TEST_DIR  = DATA_ROOT / "testing"      # patients 101-150 -> held-out test
OUT_DIR   = Path("outputs")

# --- label convention (ACDC) ---
# background=0, RV=1, MYO=2, LV=3
NUM_CLASSES = 4
LBL_RV, LBL_MYO, LBL_LV = 1, 2, 3

# --- preprocessing ---
TARGET_SPACING = (1.25, 1.25)   # mm, in-plane resample
PATCH_SIZE     = (256, 256)     # crop/pad to this after resample

# --- training ---
VAL_FRACTION = 0.15             # fraction of the 100 training patients held for val
SEED         = 42
BATCH_SIZE   = 16
MAX_EPOCHS   = 100
LR           = 1e-3
NUM_WORKERS  = 4                # set 0 on Windows if dataloader hangs

# myocardial density for mass (g/mL)
MYO_DENSITY = 1.05
