#!/usr/bin/env python3
"""Regenerate manuscript figures from saved artifacts without retraining.

Default mode regenerates figures that only read CSV artifacts. Use
``--include-model-inference`` to also regenerate MC-dropout response surfaces
from the saved fine-tuned BNN checkpoint.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def run_script(path: Path) -> None:
    print(f"\n==> {path.relative_to(PROJECT)}")
    subprocess.run([sys.executable, str(path)], cwd=str(PROJECT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manuscript figures from saved CSV/model artifacts."
    )
    parser.add_argument(
        "--include-model-inference",
        action="store_true",
        help="Also regenerate response surfaces by loading the saved BNN checkpoint.",
    )
    args = parser.parse_args()

    scripts = [
        PROJECT / "scripts/generate_fig5_bo_trajectory.py",
        PROJECT / "scripts/generate_fig8_transfer_learning.py",
    ]
    if args.include_model_inference:
        scripts.append(PROJECT / "scripts/generate_fig7_response_surfaces.py")

    for script in scripts:
        run_script(script)

    print("\nDone. Figures are in npj_ComputMater_Manuscript.assets/generated")


if __name__ == "__main__":
    main()
