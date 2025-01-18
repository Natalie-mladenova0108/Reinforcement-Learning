import gymnasium as gym
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Custom Environment
class EmptyEnv(MiniGridEnv):
    def __init__(self, size=5, agent_start_pos=(1, 1), agent_start_dir=0, max_steps=None, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(), width - 2, height - 2)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

# Parameters
MAX_STEPS = 100
EPISODES = 500
K = 10
BETA = 0.1
L = 5
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99

# Set up the environment
env = EmptyEnv(size=5)
obs_space = env.observation_space['image'].shape
action_space = env.action_space.n

# Neural Network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_size), 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.fc(x)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_network = QNetwork(obs_space, action_space).to(device)
target_network = QNetwork(obs_space, action_space).to(device)
target_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Episodic Memory
episodic_memory = []

def compute_episodic_reward(state):
    global episodic_memory
    if len(episodic_memory) == 0:
        episodic_memory.append(state)
        return 1 / (0.001 + 1)
    
    nbrs = NearestNeighbors(n_neighbors=min(K, len(episodic_memory)))
    nbrs.fit(episodic_memory)
    distances, _ = nbrs.kneighbors([state])
    d2 = np.mean(distances)
    return min(1 / (d2 + 0.001), L)

# Reward Tracking
rewards_per_episode = []

# Training Loop
for episode in range(EPISODES):
    obs, _ = env.reset()
    episodic_memory = []
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < MAX_STEPS:
        # Epsilon-greedy action selection
        if np.random.rand() < EPSILON:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(torch.tensor(obs['image'], dtype=torch.float32).unsqueeze(0).to(device))
                action = q_values.argmax().item()
        
        # Take action in the environment
        next_obs, reward, done, truncated, info = env.step(action)
        extrinsic_reward = reward
        intrinsic_reward = compute_episodic_reward(next_obs['image'].flatten())
        total_reward += extrinsic_reward + BETA * intrinsic_reward

        # Compute target Q-value
        next_q_values = q_network(torch.tensor(next_obs['image'], dtype=torch.float32).unsqueeze(0).to(device))
        next_action = next_q_values.argmax().item()

        with torch.no_grad():
            target = total_reward + DISCOUNT_FACTOR * target_network(
                torch.tensor(next_obs['image'], dtype=torch.float32).unsqueeze(0).to(device)
            )[0, next_action] * (1 - done)
        
        # Update Q-network
        predicted = q_network(torch.tensor(obs['image'], dtype=torch.float32).unsqueeze(0).to(device))[0, action]
        loss = loss_fn(predicted, target.clone().detach().to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update episodic memory and state
        episodic_memory.append(next_obs['image'].flatten())
        obs = next_obs
        steps += 1
    
    # Update target network occasionally
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())

    rewards_per_episode.append(total_reward)
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)  # Decay epsilon
    print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {EPSILON:.2f}")

# Save the Model
torch.save(q_network.state_dict(), "ngu_minigrid_empty.pth")
print("Model saved successfully!")

# Function to calculate moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Smooth the rewards
window_size = 10  # Adjust the window size for smoothing
smoothed_rewards = moving_average(rewards_per_episode, window_size)

# Plotting the rewards
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rewards_per_episode) + 1), rewards_per_episode, alpha=0.5, label="Reward per Episode", marker='o')
plt.plot(range(window_size, len(rewards_per_episode) + 1), smoothed_rewards, label=f"Smoothed Reward (window={window_size})", color='red', linewidth=2)
plt.title('Reward per Episode with Smoothed Curve')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.legend()
plt.show()
