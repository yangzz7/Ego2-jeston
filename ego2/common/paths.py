"""All project resources resolve from this checkout, never the old repository."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
MODELS = ROOT / "models"
BUILD = ROOT / "build/reconstruction"
OUTPUT = ROOT / "output"
