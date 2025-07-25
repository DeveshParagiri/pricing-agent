#!/usr/bin/env python3
"""
Main script to run RL Pricing Agent experiments.

Usage:
    python run_experiments.py --version 3 --mode train
    python run_experiments.py --version 3 --mode evaluate
    python run_experiments.py --mode test
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_training(version: int):
    """Run training for specified version."""
    script_path = f"src/training/train_ppo_v{version}.py"
    if not Path(script_path).exists():
        script_path = f"src/training/train_ppo.py"
    
    print(f"Starting training with {script_path}")
    subprocess.run([sys.executable, script_path], check=True)


def run_evaluation(version: int):
    """Run evaluation for specified version."""
    script_path = f"src/evaluation/evaluate_v{version}.py"
    if not Path(script_path).exists():
        script_path = f"src/evaluation/evaluate.py"
    
    print(f"Starting evaluation with {script_path}")
    subprocess.run([sys.executable, script_path], check=True)


def run_test():
    """Run final test comparison."""
    script_path = "src/evaluation/test_final.py"
    print(f"Running final test with {script_path}")
    subprocess.run([sys.executable, script_path], check=True)


def main():
    parser = argparse.ArgumentParser(description="Run RL Pricing Agent experiments")
    parser.add_argument("--version", type=int, default=3, choices=[1, 2, 3],
                        help="Version of the experiment to run (default: 3)")
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train", "evaluate", "test"],
                        help="Mode to run: train, evaluate, or test (default: train)")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "train":
            run_training(args.version)
        elif args.mode == "evaluate":
            run_evaluation(args.version)
        elif args.mode == "test":
            run_test()
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main() 