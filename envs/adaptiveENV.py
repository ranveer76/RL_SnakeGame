from collections import deque
import torch, math
from model.agent import device

# ------------------- Adaptive Snake Environment -------------------
class AdaptiveSnakeEnv:
    def __init__(self, num_envs=64, grid_range=(20, 200), speed_range=(5, 60)):
        self.num_envs = num_envs
        self.grid_min, self.grid_max = grid_range
        self.speed_min, self.speed_max = speed_range
        self.grid_w = self.grid_min
        self.grid_h = self.grid_min
        self.speed = self.speed_min
        self.step_counter = torch.zeros(num_envs, device=device)
        self.prev_action = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.prev_heads = [deque(maxlen=10) for _ in range(self.num_envs)]
        self.reset()

    def reset(self):
        max_len = self.grid_w * self.grid_h
        self.snake = torch.full((self.num_envs, max_len, 2), -1, device=device, dtype=torch.float32)
        self.snake[:, 0] = torch.tensor([self.grid_w // 2, self.grid_h // 2], device=device)
        self.lengths = torch.ones(self.num_envs, dtype=torch.long, device=device)
        self.direction = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32).repeat(self.num_envs, 1)
        self.food = torch.randint(0, self.grid_w, (self.num_envs, 2), device=device, dtype=torch.float32)
        self.step_counter = torch.zeros(self.num_envs, device=device)
        return self.get_state()

    def step(self, actions):
        mapping = torch.tensor([[0, -1], [0, 1], [-1, 0], [1, 0]], device=device, dtype=torch.float32)
        new_dirs = mapping[actions]
        reverse_mask = (new_dirs == -self.direction).all(dim=1)
        new_dirs[reverse_mask] = self.direction[reverse_mask]
        self.direction = new_dirs


        new_head = self.snake[:, 0, :] + self.direction
        reward = torch.full((self.num_envs,), 0.2, device=device)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        for i in range(self.num_envs):
            self.prev_heads[i].append(self.snake[i, 0].clone())
            if any(self.snake[i, 0].equal(prev) for prev in self.prev_heads[i]):
                reward[i] -= 0.2

        out_x = (new_head[:, 0] < 0) | (new_head[:, 0] >= self.grid_w)
        out_y = (new_head[:, 1] < 0) | (new_head[:, 1] >= self.grid_h)
        done |= out_x | out_y
        reward[done] = -20

        density = torch.zeros((self.num_envs,), device=device)
        for i in range(self.num_envs):
            head_i = self.snake[i, 0]
            body = self.snake[i, 1:self.lengths[i]]
            count = ((body[:, 0] >= head_i[0] - 2) & (body[:, 0] <= head_i[0] + 2) &
                    (body[:, 1] >= head_i[1] - 2) & (body[:, 1] <= head_i[1] + 2)).sum()
            density[i] = count.float() / 25.0

        for i in range(self.num_envs):
            if (self.snake[i, 1:self.lengths[i]] == new_head[i]).all(1).any():
                done[i] = True
                reward[i] = -10

        eat = (new_head == self.food).all(1)
        reward[eat] = 10 + (self.lengths[eat] / self.grid_w) ** 30.0
        for i in torch.where(eat)[0]:
            self.snake[i, 1:self.lengths[i]+1] = self.snake[i, :self.lengths[i]].clone()
            self.snake[i, 0] = new_head[i]
            self.lengths[i] += 1
            self.food[i] = torch.randint(0, self.grid_w, (2,), device=device, dtype=torch.float32)

        move = ~done & ~eat
        for i in torch.where(move)[0]:
            self.snake[i, 1:self.lengths[i]] = self.snake[i, :self.lengths[i]-1].clone()
            self.snake[i, 0] = new_head[i]

        dist = torch.norm(self.food - new_head, dim=1)
        reward += ((1.0 - dist / max(self.grid_w, self.grid_h)) ** 2) * 0.5
        reward += 0.2 * (1.0 - self.step_counter / (self.grid_w * self.grid_h))
        hunger_mask = (self.step_counter % 50 == 0)
        reward[hunger_mask] -= 1.0
        reward -= density.squeeze() * 0.5
        reward += (1.0 - density.squeeze()) * 0.3

        direction_to_food = torch.sign(self.food - new_head)
        alignment = (direction_to_food == self.direction).all(dim=1)
        reward[alignment] += 0.2


        for i in range(self.num_envs):
            if done[i]:
                self.snake[i] = -1
                self.snake[i, 0] = torch.tensor([self.grid_w // 2, self.grid_h // 2], device=device)
                self.lengths[i] = 1
                self.direction[i] = torch.tensor([1, 0], device=device)
                self.food[i] = torch.randint(0, self.grid_w, (2,), device=device, dtype=torch.float32)
                reward[i] = -0.1
                self.step_counter[i] = 0
            else:
                self.step_counter[i] += 1
                self.prev_action = actions
        reward = torch.clamp(reward, min=-10.0, max=10.0)

        return self.get_state(), reward, done

    def get_state(self):
        head = self.snake[:, 0, :]
        diff = self.food - head
        head_norm = head / torch.tensor([self.grid_w, self.grid_h], device=device)
        diff_norm = diff / torch.tensor([self.grid_w, self.grid_h], device=device)

        angle = (torch.atan2(diff[:, 1], diff[:, 0]) / math.pi + 1).unsqueeze(1) / 2.0
        norm_len = self.lengths.unsqueeze(1).float() / (self.grid_w * self.grid_h)

        grid_scale = torch.tensor([1 / self.grid_w, 1 / self.grid_h], device=device).repeat(self.num_envs, 1)
        # velocity = self.direction
        velocity = torch.zeros((self.num_envs, 4), device=device)
        dir_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
        for i in range(self.num_envs):
            dx, dy = self.direction[i].tolist()
            idx = dir_map.get((int(dx), int(dy)), 0)
            velocity[i, idx] = 1.0
        step_progress = (self.step_counter / (self.grid_w * self.grid_h)).unsqueeze(1)
        food_dist = (1.0 - torch.norm(diff, dim=1, keepdim=True) / max(self.grid_w, self.grid_h)) ** 2

        food_dir = torch.zeros((self.num_envs, 4), device=device)
        food_dir[:, 0] = (diff[:, 1] < 0).float()  # up
        food_dir[:, 1] = (diff[:, 1] > 0).float()  # down
        food_dir[:, 2] = (diff[:, 0] < 0).float()  # left
        food_dir[:, 3] = (diff[:, 0] > 0).float()  # right
        
        prev_action_onehot = torch.nn.functional.one_hot(self.prev_action, num_classes=4).float()


        danger = torch.zeros((self.num_envs, 4), device=device)
        directions = torch.tensor([[0, -1], [0, 1], [-1, 0], [1, 0]], device=device)
        for i, dir in enumerate(directions):
            next_pos = head + dir
            out_of_bounds = (
                (next_pos[:, 0] < 0) | (next_pos[:, 0] >= self.grid_w) |
                (next_pos[:, 1] < 0) | (next_pos[:, 1] >= self.grid_h)
            )
            collision = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            for j in range(self.num_envs):
                if self.lengths[j] > 1:
                    collision[j] = (self.snake[j, 1:self.lengths[j]] == next_pos[j]).all(1).any()
            danger[:, i] = (out_of_bounds | collision).float()
        
        density = torch.zeros((self.num_envs, 1), device=device)
        for i in range(self.num_envs):
            head_i = self.snake[i, 0]
            body = self.snake[i, 1:self.lengths[i]]
            count = ((body[:, 0] >= head_i[0] - 2) & (body[:, 0] <= head_i[0] + 2) &
                    (body[:, 1] >= head_i[1] - 2) & (body[:, 1] <= head_i[1] + 2)).sum()
            density[i] = count.float() / 25.0
        state = torch.cat([
            head_norm, diff_norm, food_dir, prev_action_onehot, angle, velocity, norm_len,
            grid_scale, step_progress, food_dist, danger, density
        ], dim=1)


        return state

    def scale_difficulty(self, step=0):
        self.grid_w = min(self.grid_w + 10, self.grid_max)
        self.grid_h = min(self.grid_h + 10, self.grid_max)
        self.speed = min(self.speed + 5, self.speed_max)
        print(f"📈 Difficulty scaled: Grid={self.grid_w}×{self.grid_h}, Speed={self.speed}")
        self.reset()
