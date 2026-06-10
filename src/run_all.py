"""
Convenience runner: executes all four pipeline steps in order.
Run from /workspace:  python src/run_all.py
"""
import subprocess
import sys
from pathlib import Path

STEPS = [
    "src/01_download_filter.py",
    "src/02_sentiment.py",
    "src/03_analysis.py",
    "src/04_visualize.py",
    "src/05_consistency_report.py",
]

root = Path(__file__).parent.parent

for step in STEPS:
    print(f"\n{'='*60}")
    print(f"  Running {step}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, str(root / step)], cwd=root)
    if result.returncode != 0:
        print(f"\n[ERROR] {step} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

print("\n\nAll steps complete. Charts are in /workspace/charts/")
