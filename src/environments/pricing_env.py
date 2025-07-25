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
        
        # Define observation space: [day_progress, normalized_price, recent_performance]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 10.0]),
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
        self.recent_signups = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Day progress (0 to 1)
        day_progress = self.current_day / self.days
        
        # Normalized price (0 to 1)
        normalized_price = self.current_price / self.max_price
        
        # Recent performance (average signups over last few days)
        if len(self.recent_signups) > 0:
            recent_performance = np.mean(self.recent_signups[-5:])  # Last 5 days
        else:
            recent_performance = 0.0
            
        return np.array([day_progress, normalized_price, recent_performance], dtype=np.float32)

    def step(self, action):
        # Convert action from discrete to actual price change
        action_map = {0: -1, 1: 0, 2: 1}  # 0=-1, 1=0, 2=+1
        price_action = action_map[action]
        
        # Apply price change
        price_step = 5  # Fixed price change for simplicity
        self.current_price = np.clip(self.current_price + price_action * price_step, 0, self.max_price)
        
        # Calculate expected signups based on pricing model
        signups = self.base_conversion_rate * (1 - self.current_price / self.max_price) ** self.elasticity
        
        # Add random noise to simulate market variability
        signups += np.random.normal(0, self.noise_std)
        signups = max(signups, 0)
        
        # Track recent signups for observation
        self.recent_signups.append(signups)
        
        # Revenue is price * signups
        revenue = self.current_price * signups
        self.total_revenue += revenue
        
        self.current_day += 1
        terminated = self.current_day >= self.days
        
        # Reward is the revenue for this day
        reward = revenue
        
        info = {
            "signups": signups,
            "price": self.current_price,
            "day": self.current_day,
            "total_revenue": self.total_revenue
        }
        
        return self._get_observation(), reward, terminated, False, info 