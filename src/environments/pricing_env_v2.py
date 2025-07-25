import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PricingEnv(gym.Env):
    def __init__(self, base_conversion_rate=0.3, max_price=100, elasticity=1.0, noise_std=0.05, days=30):
        super().__init__()
        
        self.base_conversion_rate = base_conversion_rate
        self.max_price = max_price
        self.elasticity = elasticity
        self.noise_std = noise_std
        self.days = days
        
        # Define action space: -1 (decrease), 0 (maintain), 1 (increase)
        self.action_space = spaces.Discrete(3)
        
        # Enhanced observation space: [day_progress, normalized_price, recent_revenue, recent_signups, price_trend]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 20.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        self.current_day = 0
        self.current_price = self.max_price / 2  # Start at mid-price
        self.total_revenue = 0
        self.recent_revenues = []
        self.recent_signups = []
        self.price_history = [self.current_price]
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Day progress (0 to 1)
        day_progress = self.current_day / self.days
        
        # Normalized price (0 to 1)
        normalized_price = self.current_price / self.max_price
        
        # Recent revenue performance (average over last 5 days)
        if len(self.recent_revenues) > 0:
            recent_revenue = np.mean(self.recent_revenues[-5:])
        else:
            recent_revenue = 0.0
            
        # Recent signup performance (average over last 5 days)
        if len(self.recent_signups) > 0:
            recent_signups = np.mean(self.recent_signups[-5:])
        else:
            recent_signups = 0.0
            
        # Price trend (-1 = decreasing, 0 = stable, 1 = increasing)
        if len(self.price_history) >= 3:
            recent_prices = self.price_history[-3:]
            if recent_prices[-1] > recent_prices[0]:
                price_trend = 1.0  # Increasing
            elif recent_prices[-1] < recent_prices[0]:
                price_trend = -1.0  # Decreasing
            else:
                price_trend = 0.0  # Stable
        else:
            price_trend = 0.0
            
        return np.array([day_progress, normalized_price, recent_revenue, recent_signups, price_trend], dtype=np.float32)

    def step(self, action):
        # Convert action from discrete to actual price change
        action_map = {0: -1, 1: 0, 2: 1}  # 0=-1, 1=0, 2=+1
        price_action = action_map[action]
        
        # Apply price change with some constraints
        price_step = 5  # Fixed price change
        new_price = self.current_price + price_action * price_step
        
        # Prevent the price from going too low (minimum $5) or too high
        self.current_price = np.clip(new_price, 5, self.max_price)
        
        # Update price history
        self.price_history.append(self.current_price)
        
        # Calculate expected signups based on pricing model
        # More realistic pricing elasticity
        price_ratio = self.current_price / self.max_price
        signups = self.base_conversion_rate * (1.0 - price_ratio) ** self.elasticity
        
        # Add random noise to simulate market variability
        signups += np.random.normal(0, self.noise_std)
        signups = max(signups, 0)
        
        # Track recent performance
        self.recent_signups.append(signups)
        
        # Calculate revenue
        revenue = self.current_price * signups
        self.recent_revenues.append(revenue)
        self.total_revenue += revenue
        
        # IMPROVED REWARD FUNCTION
        # Base reward is revenue, but with additional shaping
        base_reward = revenue
        
        # Penalty for extreme pricing strategies
        price_penalty = 0.0
        if self.current_price < 10:  # Too cheap
            price_penalty = -5.0
        elif self.current_price > 80:  # Too expensive
            price_penalty = -2.0
            
        # Bonus for maintaining reasonable revenue levels
        sustainability_bonus = 0.0
        if len(self.recent_revenues) >= 5:
            avg_recent_revenue = np.mean(self.recent_revenues[-5:])
            if avg_recent_revenue > 8.0:  # Good revenue performance
                sustainability_bonus = 2.0
                
        # Final reward combines revenue with smart pricing incentives
        reward = base_reward + price_penalty + sustainability_bonus
        
        self.current_day += 1
        terminated = self.current_day >= self.days
        
        # Bonus for good total performance at episode end
        if terminated and self.total_revenue > 200:
            reward += 10.0  # End-of-episode bonus for good total revenue
        
        info = {
            "signups": signups,
            "price": self.current_price,
            "day": self.current_day,
            "total_revenue": self.total_revenue,
            "base_reward": base_reward,
            "price_penalty": price_penalty,
            "sustainability_bonus": sustainability_bonus
        }
        
        return self._get_observation(), reward, terminated, False, info 