import gymnasium as gym
from gymnasium_minigrid.envs.empty import EmptyEnv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors

# Parameters
MAX_STEPS = 100
EPISODES = 500
K = 10
BETA = 0.1
L = 5
EPSILON = 0.1
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
        with torch.no_grad():
            target = total_reward + DISCOUNT_FACTOR * target_network(
                torch.tensor(next_obs['image'], dtype=torch.float32).unsqueeze(0).to(device)
            ).max().item() * (1 - done)
        
        # Update Q-network
        predicted = q_network(torch.tensor(obs['image'], dtype=torch.float32).unsqueeze(0).to(device))[0, action]
        loss = loss_fn(predicted, torch.tensor(target, dtype=torch.float32).to(device))

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

    print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward:.2f}")

# Save the Model
torch.save(q_network.state_dict(), "ngu_minigrid_empty.pth")
print("Model saved successfully!")
