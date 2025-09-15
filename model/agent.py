# agent.py
import torch, copy, random
import torch.nn as nn
import torch.optim as optim
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- DQN Agent -------------------
class DQNAgent:
    def __init__(self, state_size=27, action_size=4, memory_size=100_000, neurons=128, gamma=0.99, epsilon=1.0, epsilon_decay=0.997, epsilon_min=0.01, lr=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr

        self.model = self._build_model(neurons).to(device)
        self.target_model= copy.deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.update_target()

    def _build_model(self, neurons):
        return nn.Sequential(
            nn.Linear(self.state_size, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, self.action_size)
        )

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, states):
        if random.random() < self.epsilon:
            return torch.randint(0, self.action_size, (states.size(0),), device=device)
        with torch.no_grad():
            q_values = self.model(states)
            return torch.argmax(q_values, dim=1)

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def replay(self, batch_size=1024):
        if len(self.memory) < batch_size:
            print("oom")
            return
        batch = random.sample(self.memory, batch_size)
        s, a, r, ns, d = zip(*batch)
        states = torch.stack(s).to(device)
        next_states = torch.stack(ns).to(device)
        actions = torch.LongTensor(a).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(r).to(device)
        dones = torch.FloatTensor(d).to(device)

        current_q = self.model(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), dim=1, keepdim=True)
            max_next_q = self.target_model(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def soft_update_target(self, tau=0.005):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path="snake_dqn.pth"):
        torch.save({
            'model': self.model.state_dict(),
            'target': self.target_model.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"✅ Model saved to {path}")

    def load(self, path="snake_dqn.pth"):
        import os
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            self.model.load_state_dict(checkpoint['model'])
            self.target_model.load_state_dict(checkpoint['target'])
            self.epsilon = checkpoint.get('epsilon', 1.0)
            self.model.eval()
            self.target_model.eval()
            print(f"✅ Model loaded from {path}")
