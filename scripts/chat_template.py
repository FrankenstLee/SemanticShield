"""Compatibility wrapper for converting prompts to the Qwen chat template."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from semanticshield.training.chat_template import main


if __name__ == "__main__":
    main()


