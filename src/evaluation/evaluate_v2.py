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

from src.environments.pricing_env_v2 import PricingEnv


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

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.sample()


def run_episode(env, agent=None, random_policy=False):
    """Run a single episode and return detailed statistics"""
    obs, _ = env.reset()
    total_reward = 0
    done = False
    
    # Track daily statistics
    daily_stats = {
        'days': [],
        'prices': [],
        'signups': [],
        'revenues': [],
        'actions': [],
        'price_penalties': [],
        'sustainability_bonuses': []
    }
    
    step = 0
    while not done:
        if random_policy:
            # Random action: 0, 1, or 2 (decrease, maintain, increase)
            action = np.random.choice([0, 1, 2])
        else:
            # Use trained agent
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = agent.get_action(obs_tensor).item()
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Record daily stats (including new reward components)
        daily_stats['days'].append(step + 1)
        daily_stats['prices'].append(info['price'])
        daily_stats['signups'].append(info['signups'])
        daily_stats['revenues'].append(info['base_reward'])  # Base revenue without bonuses
        daily_stats['actions'].append(action)
        daily_stats['price_penalties'].append(info.get('price_penalty', 0))
        daily_stats['sustainability_bonuses'].append(info.get('sustainability_bonus', 0))
        
        step += 1
    
    return total_reward, daily_stats


def evaluate_agents(num_episodes=10):
    """Evaluate both random and trained agents"""
    
    # Load trained agent if available
    env = PricingEnv()
    trained_agent = None
    
    try:
        trained_agent = Agent(env.observation_space, env.action_space)
        model_path = project_root / "models" / "ppo_pricing_agent_v2.pth"
        trained_agent.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        trained_agent.eval()
        print("✅ Loaded trained PPO agent v2 (improved)")
    except FileNotFoundError:
        print("❌ No trained agent v2 found. Please run 'python train_ppo_v2.py' first.")
        return
    
    # Test both agents
    print(f"\n🔬 Evaluating agents over {num_episodes} episodes...")
    
    # Random agent results
    random_rewards = []
    random_stats = []
    
    # Trained agent results
    trained_rewards = []
    trained_stats = []
    
    for episode in range(num_episodes):
        # Test random agent
        env = PricingEnv()
        reward, stats = run_episode(env, random_policy=True)
        random_rewards.append(reward)
        if episode == 0:  # Save first episode for visualization
            random_stats = stats
        
        # Test trained agent
        env = PricingEnv()
        reward, stats = run_episode(env, agent=trained_agent, random_policy=False)
        trained_rewards.append(reward)
        if episode == 0:  # Save first episode for visualization
            trained_stats = stats
    
    # Calculate statistics
    random_mean = np.mean(random_rewards)
    random_std = np.std(random_rewards)
    trained_mean = np.mean(trained_rewards)
    trained_std = np.std(trained_rewards)
    
    improvement = ((trained_mean - random_mean) / random_mean) * 100
    
    print(f"\n📊 Results over {num_episodes} episodes:")
    print(f"Random Agent:  ${random_mean:.2f} ± ${random_std:.2f}")
    print(f"Trained Agent: ${trained_mean:.2f} ± ${trained_std:.2f}")
    print(f"Improvement:   {improvement:.1f}%")
    
    # Create visualization
    create_comparison_plot(random_stats, trained_stats, random_mean, trained_mean)
    
    return {
        'random': {'mean': random_mean, 'std': random_std, 'rewards': random_rewards},
        'trained': {'mean': trained_mean, 'std': trained_std, 'rewards': trained_rewards},
        'improvement': improvement
    }


def create_comparison_plot(random_stats, trained_stats, random_mean, trained_mean):
    """Create a comprehensive comparison plot"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🚀 Improved PPO Agent vs Random Policy', fontsize=16, fontweight='bold')
    
    days = random_stats['days']
    
    # Plot 1: Daily Prices
    ax1.plot(days, random_stats['prices'], 'r-', alpha=0.7, linewidth=2, label='Random Policy')
    ax1.plot(days, trained_stats['prices'], 'b-', alpha=0.7, linewidth=2, label='PPO Agent v2')
    ax1.set_title('Daily Pricing Strategy')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Min Price ($5)')
    
    # Plot 2: Daily Signups
    ax2.plot(days, random_stats['signups'], 'r-', alpha=0.7, linewidth=2, label='Random Policy')
    ax2.plot(days, trained_stats['signups'], 'b-', alpha=0.7, linewidth=2, label='PPO Agent v2')
    ax2.set_title('Daily Signups')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Signups')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Daily Revenue (Base + Bonuses)
    ax3.plot(days, random_stats['revenues'], 'r-', alpha=0.7, linewidth=2, label='Random Policy')
    ax3.plot(days, trained_stats['revenues'], 'b-', alpha=0.7, linewidth=2, label='PPO Agent v2')
    ax3.set_title('Daily Base Revenue')
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Revenue ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Total Revenue Comparison
    categories = ['Random Policy', 'PPO Agent v2']
    total_revenues = [random_mean, trained_mean]
    colors = ['red', 'blue']
    
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
    ax4.text(0.5, max(total_revenues) * 0.8, f'Improvement: {improvement:.1f}%', 
             ha='center', transform=ax4.transData, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    chart_path = project_root / "results" / "pricing_comparison_v2.png"
    plt.savefig(str(chart_path), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📈 Visualization saved as '{chart_path}'")
    
    # Print reward breakdown for trained agent
    print(f"\n🔍 Reward Breakdown (Trained Agent):")
    print(f"Base Revenue: ${np.sum(trained_stats['revenues']):.2f}")
    print(f"Price Penalties: ${np.sum(trained_stats['price_penalties']):.2f}")
    print(f"Sustainability Bonuses: ${np.sum(trained_stats['sustainability_bonuses']):.2f}")


def test_single_episode():
    """Quick test with a single episode for debugging"""
    print("🧪 Running single episode test with improved environment...")
    
    # Test with random policy
    env = PricingEnv()
    total_reward, stats = run_episode(env, random_policy=True)
    print(f"Random policy total reward: ${total_reward:.2f}")
    
    # Test with trained agent (if available)
    try:
        env = PricingEnv()
        agent = Agent(env.observation_space, env.action_space)
        model_path = project_root / "models" / "ppo_pricing_agent_v2.pth"
        agent.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        agent.eval()
        
        total_reward, stats = run_episode(env, agent=agent, random_policy=False)
        print(f"Trained agent v2 total reward: ${total_reward:.2f}")
        
        # Show pricing strategy
        min_price = min(stats['prices'])
        max_price = max(stats['prices'])
        avg_price = np.mean(stats['prices'])
        print(f"Price range: ${min_price:.0f} - ${max_price:.0f} (avg: ${avg_price:.1f})")
        
    except FileNotFoundError:
        print("No trained agent v2 found - run training first!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_single_episode()
    else:
        evaluate_agents(num_episodes=10) 