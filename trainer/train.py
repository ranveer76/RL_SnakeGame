import torch, os
from model.agent import DQNAgent
from envs.adaptiveENV import AdaptiveSnakeEnv

# ------------------- Training Loop -------------------
def train_adaptive(minEPS=0.2, EPSMULT=1.5, model = "model.pth", NUM_ENVS=64, BATCH_SIZE = 128, SAVE_EVERY = 500, TARGET_UPDATE = 250, WARMUP = 5000, ONLINE_ADAPT = True, state_size=27, action_size=4, neurons=128, AVGLEN=20, LOGFREQ=10, QLOGFREQ=100, SCALEFREQ=1000):
    agent = DQNAgent(state_size, action_size, neurons)
    
    if os.path.exists(model):
        agent.load(model)
        agent.epsilon = agent.epsilon or 1.0  # fallback if missing
        agent.epsilon = min(minEPS, agent.epsilon * EPSMULT)

    env = AdaptiveSnakeEnv(num_envs=NUM_ENVS)
    states = env.reset()
    step = 0

    while True:
        step += 1
        actions = agent.act(states)
        next_states, rewards, done = env.step(actions)

        for i in range(NUM_ENVS):
            agent.remember(states[i], actions[i], rewards[i], next_states[i], done[i])

        states = next_states

        if ONLINE_ADAPT or step > WARMUP:
            agent.replay(batch_size=BATCH_SIZE)

        if step % TARGET_UPDATE == 0:
            agent.update_target()
        else:
            agent.soft_update_target()
        if step % SAVE_EVERY == 0:
            agent.save(model)

        avg_len = env.lengths.float().mean().item()
        if step % LOGFREQ == 0:
            avg_reward = rewards.mean().item()
            max_len = env.lengths.max().item()
            print(f"[Step {step}] ε={agent.epsilon:.3f} | AvgR={avg_reward:.2f} | MaxLen={max_len} | AvgLen={avg_len:.2f}")
        
        if step % QLOGFREQ == 0:
            with torch.no_grad():
                q_vals = agent.model(states)
                avg_q = q_vals.mean().item()
            print(f"[Step {step}] AvgQ={avg_q:.2f}")

        if avg_len > AVGLEN and step % SCALEFREQ == 0:
            env.scale_difficulty(step)
            agent.epsilon = min(minEPS, agent.epsilon * EPSMULT)

if __name__ == "__main__":
    train_adaptive(model=("../models/"+input("model name: ")))
