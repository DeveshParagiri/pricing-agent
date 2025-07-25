import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
import gymnasium as gym

from src.environments.pricing_env_v3 import PricingEnv


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space.n), std=0.01),
        )

    def get_action(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.sample()


def run_episode(env, agent=None, random_policy=False):
    """Run a single episode and return detailed statistics"""
    obs, _ = env.reset()
    total_reward = 0
    done = False
    
    daily_stats = {
        'days': [],
        'prices': [],
        'signups': [],
        'revenues': [],
        'actions': []
    }
    
    step = 0
    while not done:
        if random_policy:
            action = np.random.choice([0, 1, 2])
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = agent.get_action(obs_tensor).item()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += info['base_reward']  # Use base revenue only
        
        daily_stats['days'].append(step + 1)
        daily_stats['prices'].append(info['price'])
        daily_stats['signups'].append(info['signups'])
        daily_stats['revenues'].append(info['base_reward'])
        daily_stats['actions'].append(action)
        
        step += 1
    
    return total_reward, daily_stats


def evaluate_agents(num_episodes=10):
    """Evaluate both random and trained agents"""
    
    env = PricingEnv()
    trained_agent = None
    
    try:
        trained_agent = Agent(env.observation_space, env.action_space)
        model_path = project_root / "models" / "ppo_pricing_agent_v3.pth"
        trained_agent.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        trained_agent.eval()
        print("✅ Loaded trained PPO agent v3")
    except FileNotFoundError:
        print("❌ No trained agent v3 found. Please run 'python train_ppo_v3.py' first.")
        return
    
    print(f"\n🔬 Evaluating agents over {num_episodes} episodes...")
    
    random_rewards = []
    random_stats = []
    trained_rewards = []
    trained_stats = []
    
    for episode in range(num_episodes):
        # Test random agent
        env = PricingEnv()
        reward, stats = run_episode(env, random_policy=True)
        random_rewards.append(reward)
        if episode == 0:
            random_stats = stats
        
        # Test trained agent
        env = PricingEnv()
        reward, stats = run_episode(env, agent=trained_agent, random_policy=False)
        trained_rewards.append(reward)
        if episode == 0:
            trained_stats = stats
    
    random_mean = np.mean(random_rewards)
    random_std = np.std(random_rewards)
    trained_mean = np.mean(trained_rewards)
    trained_std = np.std(trained_rewards)
    
    improvement = ((trained_mean - random_mean) / random_mean) * 100
    
    print(f"\n📊 Results over {num_episodes} episodes:")
    print(f"Random Agent:  ${random_mean:.2f} ± ${random_std:.2f}")
    print(f"Trained Agent: ${trained_mean:.2f} ± ${trained_std:.2f}")
    print(f"Improvement:   {improvement:.1f}%")
    
    create_comparison_plot(random_stats, trained_stats, random_mean, trained_mean)
    
    return {
        'random': {'mean': random_mean, 'std': random_std},
        'trained': {'mean': trained_mean, 'std': trained_std},
        'improvement': improvement
    }


def create_comparison_plot(random_stats, trained_stats, random_mean, trained_mean):
    """Create final comparison plot"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Final Results: PPO Agent v3 vs Random Policy', fontsize=16, fontweight='bold')
    
    days = random_stats['days']
    
    # Plot 1: Daily Prices
    ax1.plot(days, random_stats['prices'], 'r-', alpha=0.7, linewidth=2, label='Random Policy')
    ax1.plot(days, trained_stats['prices'], 'b-', alpha=0.7, linewidth=2, label='PPO Agent v3')
    ax1.set_title('Daily Pricing Strategy')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Theoretical Optimal ($50)')
    
    # Plot 2: Daily Signups
    ax2.plot(days, random_stats['signups'], 'r-', alpha=0.7, linewidth=2, label='Random Policy')
    ax2.plot(days, trained_stats['signups'], 'b-', alpha=0.7, linewidth=2, label='PPO Agent v3')
    ax2.set_title('Daily Signups')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Signups')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Daily Revenue
    ax3.plot(days, random_stats['revenues'], 'r-', alpha=0.7, linewidth=2, label='Random Policy')
    ax3.plot(days, trained_stats['revenues'], 'b-', alpha=0.7, linewidth=2, label='PPO Agent v3')
    ax3.set_title('Daily Revenue')
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Revenue ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Total Revenue Comparison
    categories = ['Random Policy', 'PPO Agent v3', 'Theoretical Optimal']
    total_revenues = [random_mean, trained_mean, 229.22]  # Include theoretical optimal
    colors = ['red', 'blue', 'green']
    
    bars = ax4.bar(categories, total_revenues, color=colors, alpha=0.7)
    ax4.set_title('Total Revenue Comparison')
    ax4.set_ylabel('Total Revenue ($)')
    
    # Add value labels on bars
    for bar, value in zip(bars, total_revenues):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Calculate improvement percentage
    improvement = ((trained_mean - random_mean) / random_mean) * 100
    color = 'lightgreen' if improvement > 0 else 'lightcoral'
    ax4.text(1, max(total_revenues) * 0.8, f'Improvement: {improvement:.1f}%', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    chart_path = project_root / "results" / "final_pricing_comparison_v3.png"
    plt.savefig(str(chart_path), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📈 Final visualization saved as '{chart_path}'")


if __name__ == "__main__":
    evaluate_agents(num_episodes=10) 