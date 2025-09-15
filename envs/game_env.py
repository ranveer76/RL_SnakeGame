# game_env.py

import torch
from .adaptiveENV import AdaptiveSnakeEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SingleSnakeWrapper:
    def __init__(self):
        self.env = AdaptiveSnakeEnv(num_envs=1)
    
    def reset(self):
        state = self.env.reset()
        return state[0]  # extract single state

    def step(self, action):
        actions = torch.tensor([action], device=device)
        next_states, rewards, dones = self.env.step(actions)
        return next_states[0], rewards[0].item(), dones[0].item()

    def get_snake(self):
        return self.env.snake[0, :self.env.lengths[0]].cpu().numpy()

    def get_food(self):
        return self.env.food[0].cpu().numpy()
