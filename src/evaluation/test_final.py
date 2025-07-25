import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

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


def test_agents():
    """Test both random and trained agents"""
    print("🧪 Testing Final Agents - V3 Environment")
    print("=" * 50)
    
    # Test random agent
    print("\n🎲 Testing Random Agent...")
    random_rewards = []
    for episode in range(5):
        env = PricingEnv()
        obs, _ = env.reset()
        total_reward = 0
        prices = []
        
        for step in range(30):
            action = np.random.choice([0, 1, 2])  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += info['base_reward']  # Use base revenue only
            prices.append(info['price'])
            
            if terminated or truncated:
                break
        
        random_rewards.append(total_reward)
        print(f"  Episode {episode+1}: ${total_reward:.2f}, Price range: ${min(prices):.0f}-${max(prices):.0f}")
    
    random_avg = np.mean(random_rewards)
    print(f"Random Agent Average: ${random_avg:.2f}")
    
    # Test trained agent
    print("\n🤖 Testing Trained Agent...")
    
    try:
        env = PricingEnv()
        agent = Agent(env.observation_space, env.action_space)
        model_path = project_root / "models" / "ppo_pricing_agent_v3.pth"
        agent.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        agent.eval()
        
        trained_rewards = []
        for episode in range(5):
            env = PricingEnv()
            obs, _ = env.reset()
            total_reward = 0
            prices = []
            revenues = []
            
            for step in range(30):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action = agent.get_action(obs_tensor).item()
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += info['base_reward']  # Use base revenue only
                prices.append(info['price'])
                revenues.append(info['base_reward'])
                
                if terminated or truncated:
                    break
            
            trained_rewards.append(total_reward)
            print(f"  Episode {episode+1}: ${total_reward:.2f}, Price range: ${min(prices):.0f}-${max(prices):.0f}, Avg price: ${np.mean(prices):.1f}")
        
        trained_avg = np.mean(trained_rewards)
        improvement = ((trained_avg - random_avg) / random_avg) * 100
        
        print(f"Trained Agent Average: ${trained_avg:.2f}")
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Random Agent:  ${random_avg:.2f}")
        print(f"Trained Agent: ${trained_avg:.2f}")
        print(f"Improvement:   {improvement:.1f}%")
        
        if improvement > 0:
            print("✅ SUCCESS: Trained agent outperforms random policy!")
        else:
            print("❌ FAILURE: Trained agent underperforms")
        
        # Show optimal theoretical performance
        print(f"\n📊 Analysis:")
        print(f"- Random pricing typically gets ~$150-250")
        print(f"- Optimal pricing (around $40-50) should get ~$280-320")
        print(f"- Your agent achieved: ${trained_avg:.2f}")
        
    except FileNotFoundError:
        print("❌ No trained agent found!")


def find_optimal_price():
    """Find the theoretical optimal price"""
    print("\n🔍 Finding Theoretical Optimal Price...")
    
    env = PricingEnv()
    best_revenue = 0
    best_price = 0
    
    for test_price in range(10, 101, 5):
        total_revenue = 0
        for trial in range(10):  # Average over multiple trials
            env = PricingEnv()
            obs, _ = env.reset()
            
            # Force specific price for all 30 days
            for day in range(30):
                env.current_price = test_price
                # Calculate signups manually
                price_ratio = test_price / env.max_price
                signups = env.base_conversion_rate * (1.0 - price_ratio) ** env.elasticity
                signups += np.random.normal(0, env.noise_std)
                signups = max(signups, 0)
                revenue = test_price * signups
                total_revenue += revenue
        
        avg_revenue = total_revenue / 10
        if avg_revenue > best_revenue:
            best_revenue = avg_revenue
            best_price = test_price
    
    print(f"Theoretical optimal price: ${best_price}")
    print(f"Theoretical optimal revenue: ${best_revenue:.2f}")


if __name__ == "__main__":
    test_agents()
    find_optimal_price() 