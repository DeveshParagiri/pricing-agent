# RL Pricing Agent

A reinforcement learning project that trains PPO agents to learn optimal pricing strategies for a SaaS product.

## Overview

This project implements three versions of a PPO (Proximal Policy Optimization) agent designed to learn pricing strategies in a simulated market environment. The agents interact with different reward structures and penalty systems to optimize revenue generation.

## 🛠️ Project Structure

```
refactor/
├── src/
│   ├── environments/
│   │   ├── pricing_env.py         # Original environment
│   │   ├── pricing_env_v2.py      # Version 2 with penalties
│   │   └── pricing_env_v3.py      # Version 3 with comprehensive rewards
│   ├── training/
│   │   ├── train_ppo.py           # Original PPO training
│   │   ├── train_ppo_v2.py        # Version 2 training
│   │   └── train_ppo_v3.py        # Version 3 training
│   └── evaluation/
│       ├── evaluate.py            # Original evaluation
│       ├── evaluate_v2.py         # Version 2 evaluation
│       ├── evaluate_v3.py         # Version 3 evaluation
│       └── test_final.py          # Clean testing script
├── models/
│   ├── ppo_pricing_agent.pth      # Trained model v1
│   ├── ppo_pricing_agent_v2.pth   # Trained model v2
│   └── ppo_pricing_agent_v3.pth   # Trained model v3
├── results/
│   ├── pricing_comparison.png           # V1 results
│   ├── pricing_comparison_v2.png        # V2 results
│   └── final_pricing_comparison_v3.png  # V3 results
├── run_experiments.py     # Main script to run experiments
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone the repository
2. Create a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate

   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Quick Start

Use the main experiment runner (recommended):

```bash
python run_all_experiments.py
```

```bash
# Train the latest version (v3)
python run_experiments.py --version 3 --mode train

# Evaluate with visualization
python run_experiments.py --version 3 --mode evaluate

# Run final comparison test
python run_experiments.py --mode test
```

### Run Individual Scripts

You can also run scripts directly:

```bash
# Train different versions
python src/training/train_ppo_v3.py    # Latest version
python src/training/train_ppo_v2.py    # Version 2
python src/training/train_ppo.py       # Original version

# Evaluate models
python src/evaluation/evaluate_v3.py   # Evaluate v3 with charts
python src/evaluation/test_final.py    # Clean comparison
```

### Command Line Arguments

The main experiment runner supports:

- `--version {1,2,3}`: Choose which version to run (default: 3)
- `--mode {train,evaluate,test}`: Operation mode (default: train)

## 🎯 Environment Details

### Pricing Environment

The environment simulates a SaaS pricing scenario with:

- **Actions**: Decrease price (-$5), Maintain, Increase price (+$5)
- **State Space**: Current price, revenue history, market conditions
- **Reward**: Revenue-based with various penalty/bonus systems across versions
- **Episode Length**: 30 days of simulation

### Version Differences

- **v1**: Simple revenue maximization
- **v2**: Added penalty system for extreme pricing
- **v3**: Comprehensive reward design with efficiency bonuses

## 📈 Model Training

All models use PPO with:

- Learning rate: 5e-4
- 30,000 total timesteps
- GAE lambda: 0.95
- Clip coefficient: 0.2

Trained models are automatically saved in the `models/` directory.

## 🔍 Results

Training results and comparison charts are saved in the `results/` directory. Each version generates visualization comparing agent performance against random pricing policies.

## 📋 Dependencies

Core libraries:

- `gymnasium` - RL environment framework
- `torch` - Neural network training
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `cleanrl` - PPO implementation base

See `requirements.txt` for complete dependency list with versions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
