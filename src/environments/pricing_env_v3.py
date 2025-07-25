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
        
        # Enhanced observation space: [day_progress, normalized_price, recent_revenue, recent_signups, revenue_trend]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 50.0, 1.0, 1.0]),
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
        self.revenue_history = []
        
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
            
        # Revenue trend (-1 = decreasing, 0 = stable, 1 = increasing)
        if len(self.revenue_history) >= 3:
            recent_revenues = self.revenue_history[-3:]
            if recent_revenues[-1] > recent_revenues[0] * 1.05:  # 5% increase threshold
                revenue_trend = 1.0  # Increasing
            elif recent_revenues[-1] < recent_revenues[0] * 0.95:  # 5% decrease threshold
                revenue_trend = -1.0  # Decreasing
            else:
                revenue_trend = 0.0  # Stable
        else:
            revenue_trend = 0.0
            
        return np.array([day_progress, normalized_price, recent_revenue, recent_signups, revenue_trend], dtype=np.float32)

    def step(self, action):
        # Convert action from discrete to actual price change
        action_map = {0: -1, 1: 0, 2: 1}  # 0=-1, 1=0, 2=+1
        price_action = action_map[action]
        
        # Apply price change
        price_step = 5  # Fixed price change
        new_price = self.current_price + price_action * price_step
        
        # Prevent extreme pricing (minimum $10, maximum $100)
        self.current_price = np.clip(new_price, 10, self.max_price)
        
        # Update price history
        self.price_history.append(self.current_price)
        
        # Calculate expected signups based on pricing model
        # Optimal price point is around $40-60 for maximum revenue
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
        self.revenue_history.append(revenue)
        self.total_revenue += revenue
        
        # MUCH IMPROVED REWARD FUNCTION
        # The goal is to find the optimal price point that maximizes revenue
        
        # Base reward is revenue
        base_reward = revenue
        
        # STRONG penalty for sub-optimal pricing
        price_efficiency_bonus = 0.0
        optimal_price_range = (30, 70)  # Sweet spot for revenue
        
        if optimal_price_range[0] <= self.current_price <= optimal_price_range[1]:
            # Reward for being in the optimal range
            price_efficiency_bonus = 5.0
        elif self.current_price < 25:
            # Heavy penalty for underpricing
            price_efficiency_bonus = -15.0
        elif self.current_price > 80:
            # Moderate penalty for overpricing
            price_efficiency_bonus = -8.0
        
        # Revenue consistency bonus
        consistency_bonus = 0.0
        if len(self.recent_revenues) >= 5:
            avg_recent_revenue = np.mean(self.recent_revenues[-5:])
            revenue_std = np.std(self.recent_revenues[-5:])
            
            # Bonus for high average revenue
            if avg_recent_revenue > 12.0:
                consistency_bonus += 3.0
            
            # Bonus for consistent revenue (low variance)
            if revenue_std < 2.0 and avg_recent_revenue > 8.0:
                consistency_bonus += 2.0
                
        # Trend bonus - reward improving revenue
        trend_bonus = 0.0
        if len(self.revenue_history) >= 3:
            recent_trend = np.mean(self.revenue_history[-3:]) - np.mean(self.revenue_history[-6:-3]) if len(self.revenue_history) >= 6 else 0
            if recent_trend > 0.5:  # Revenue is improving
                trend_bonus = 2.0
            elif recent_trend < -1.0:  # Revenue is declining
                trend_bonus = -3.0
        
        # Final reward combines all factors
        reward = base_reward + price_efficiency_bonus + consistency_bonus + trend_bonus
        
        self.current_day += 1
        terminated = self.current_day >= self.days
        
        # Strong end-of-episode bonus for good total performance
        if terminated:
            if self.total_revenue > 350:  # Very good performance
                reward += 20.0
            elif self.total_revenue > 280:  # Good performance
                reward += 10.0
            elif self.total_revenue < 200:  # Poor performance
                reward -= 15.0
        
        info = {
            "signups": signups,
            "price": self.current_price,
            "day": self.current_day,
            "total_revenue": self.total_revenue,
            "base_reward": base_reward,
            "price_efficiency_bonus": price_efficiency_bonus,
            "consistency_bonus": consistency_bonus,
            "trend_bonus": trend_bonus
        }
        
        return self._get_observation(), reward, terminated, False, info 