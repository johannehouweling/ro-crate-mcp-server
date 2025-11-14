# Ensure the repository root is on sys.path so tests can import the package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
