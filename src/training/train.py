from pricing_env import PricingEnv
import numpy as np


def run_episode(env):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        # For now, choose a random action: -1, 0, or 1
        action = np.random.choice([-1, 0, 1])
        state, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward


if __name__ == "__main__":
    env = PricingEnv()
    total_reward = run_episode(env)
    print("Episode finished with total reward:", total_reward) 