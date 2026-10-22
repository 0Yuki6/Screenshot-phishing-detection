"""
Task runner CLI for Phishing Screenshot Detector.
Usage:
    python tasks.py prepare
    python tasks.py train
    python tasks.py evaluate
    python tasks.py test
    python tasks.py reproduce
    python tasks.py api
    python tasks.py web
"""

import sys
import subprocess
import os


def run_command(cmd: str, cwd: str = "."):
    print(f"\n[RUNNING] {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        print(f"[FAILED] Command exited with code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[SUCCESS] {cmd}\n")


def prepare():
    """Prepare datasets, strip chrome, compute group IDs, and generate supplement."""
    print("=== Step 1: Dataset preparation and grouping ===")
    run_command("python -m src.dataset")
    print("=== Step 2: Generating supplementary hard samples ===")
    run_command("python -m src.generate_supplement")


def test():
    """Run full test suite."""
    print("=== Running pytest test suite ===")
    run_command("python -m pytest tests/ -v")


def train():
    """Train the model with group-aware cross-validation and probability calibration."""
    print("=== Training model ===")
    run_command("python -m src.train")


def evaluate():
    """Run full evaluation suite with 95% Wilson CIs, risk-coverage curves, and error diagnostics."""
    print("=== Running evaluation ===")
    run_command("python -m src.evaluate")


def reproduce():
    """Run end-to-end pipeline: prepare -> train -> evaluate -> test."""
    print("==================================================")
    print(" REPRODUCING PIPELINE END-TO-END (SEED 42)")
    print("==================================================")
    prepare()
    train()
    evaluate()
    test()
    print("==================================================")
    print(" REPRODUCE COMPLETED SUCCESSFULLY")
    print("==================================================")


def api():
    """Start FastAPI server."""
    print("=== Starting FastAPI server on http://localhost:8000 ===")
    run_command("python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")


def web():
    """Start React frontend."""
    client_dir = "client use" if os.path.exists("client use") else "web"
    print(f"=== Starting Vite frontend in {client_dir} ===")
    run_command("npm run dev", cwd=client_dir)


def main():
    if len(sys.argv) < 2:
        print("Usage: python tasks.py [prepare|train|evaluate|test|reproduce|api|web]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    commands = {
        "prepare": prepare,
        "train": train,
        "evaluate": evaluate,
        "test": test,
        "reproduce": reproduce,
        "api": api,
        "web": web,
    }

    if cmd in commands:
        commands[cmd]()
    else:
        print(f"Unknown command: '{cmd}'. Available: {list(commands.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
