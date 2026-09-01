#!/usr/bin/env python3
"""Run the deterministic synthetic surveyor maturity proof."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from encuestador.transport_proof import run_synthetic_proof  # noqa: E402


def main() -> int:
    summary = run_synthetic_proof()
    if summary["status"] != "synthetic_proof_only":
        raise SystemExit("unexpected proof status")
    if summary["fold_policy"]["households_crossing_folds"] != 0:
        raise SystemExit("household fold leakage")
    if summary["weight_policy"]["fit"] is not None:
        raise SystemExit("unexpected fit weight")
    if summary["temporal_policy"]["target_period_calibration"] != "none":
        raise SystemExit("unexpected target-period calibration")
    if summary["semantic_plane"]["real_vintage_approval"] is not False:
        raise SystemExit("synthetic proof claimed real-vintage approval")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
