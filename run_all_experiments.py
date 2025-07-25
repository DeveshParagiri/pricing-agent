#!/usr/bin/env python3
"""
Comprehensive script to run all PPO pricing agent experiments.
Trains and evaluates all 3 versions sequentially.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 50)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully in {duration:.1f}s")
    else:
        print(f"❌ {description} failed with exit code {result.returncode}")
        return False
    return True

def main():
    print("🎯 Starting Complete PPO Pricing Agent Experiment Pipeline")
    print("📊 This will train and evaluate all 3 versions sequentially")
    print("\n" + "="*60)
    
    # Check if we're in the virtual environment
    venv_python = Path("env/bin/python")
    if venv_python.exists():
        python_cmd = str(venv_python)
        print("🐍 Using virtual environment Python")
    else:
        python_cmd = sys.executable
        print("🐍 Using system Python (consider activating virtual environment)")
    
    # Define the experiment sequence
    experiments = [
        # Version 1
        ([python_cmd, "run_experiments.py", "--version", "1", "--mode", "train"], 
         "Training PPO Agent v1 (Basic Revenue Optimization)"),
        ([python_cmd, "run_experiments.py", "--version", "1", "--mode", "evaluate"], 
         "Evaluating PPO Agent v1 with Visualizations"),
        
        # Version 2  
        ([python_cmd, "run_experiments.py", "--version", "2", "--mode", "train"], 
         "Training PPO Agent v2 (With Penalty System)"),
        ([python_cmd, "run_experiments.py", "--version", "2", "--mode", "evaluate"], 
         "Evaluating PPO Agent v2 with Visualizations"),
        
        # Version 3
        ([python_cmd, "run_experiments.py", "--version", "3", "--mode", "train"], 
         "Training PPO Agent v3 (Comprehensive Rewards)"),
        ([python_cmd, "run_experiments.py", "--version", "3", "--mode", "evaluate"], 
         "Evaluating PPO Agent v3 with Visualizations"),
        
        # Final comparison
        ([python_cmd, "run_experiments.py", "--mode", "test"], 
         "Running Final Cross-Version Comparison")
    ]
    
    total_experiments = len(experiments)
    completed = 0
    
    for i, (cmd, description) in enumerate(experiments, 1):
        print(f"\n📍 Step {i}/{total_experiments}")
        
        if run_command(cmd, description):
            completed += 1
        else:
            print(f"\n💥 Pipeline failed at step {i}. Stopping execution.")
            break
        
        # Small delay between experiments
        if i < total_experiments:
            print(f"\n⏱️  Pausing 2 seconds before next step...")
            time.sleep(2)
    
    # Summary
    print("\n" + "="*60)
    print("📋 EXPERIMENT PIPELINE SUMMARY")
    print("="*60)
    print(f"✅ Completed: {completed}/{total_experiments} experiments")
    
    if completed == total_experiments:
        print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("\n📁 Check these directories for results:")
        print("   📁 models/     - Trained agent files (.pth)")
        print("   📁 results/    - Comparison charts (.png)")
        print("\n🔍 Key files generated:")
        print("   • ppo_pricing_agent.pth      (v1 model)")
        print("   • ppo_pricing_agent_v2.pth   (v2 model)")  
        print("   • ppo_pricing_agent_v3.pth   (v3 model)")
        print("   • pricing_comparison.png     (v1 results)")
        print("   • pricing_comparison_v2.png  (v2 results)")
        print("   • final_pricing_comparison_v3.png (v3 results)")
    else:
        print(f"\n⚠️  Pipeline incomplete. {total_experiments - completed} experiments failed.")
        print("💡 Try running individual experiments with:")
        print("   python run_experiments.py --version <1|2|3> --mode <train|evaluate>")
    
    print("\n🏁 Pipeline execution finished.")

if __name__ == "__main__":
    main() 