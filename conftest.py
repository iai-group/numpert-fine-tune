import sys
from pathlib import Path

# Ensure the project root is on sys.path so that 'main' and 'code.*'
# are importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
