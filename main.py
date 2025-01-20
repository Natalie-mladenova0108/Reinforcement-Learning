import gymnasium as gym
from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, FlatObsWrapper, RGBImgObsWrapper
from gymnasium.wrappers.filter_observation import FilterObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import random
from gymnasium import spaces

# Parameters
SIZE = 5
MAX_STEPS = 50
EPISODES = 500
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 1
REPLAY_BUFFER = 10000
BATCH_SIZE = 32
TARGET_NETWORK_UPDATES = 25

class EnvWrapper(gym.Wrapper):
    """
    For now, this just minimizes the action space of the environment.
    (It is set to 7, although 4 of them are unused)
    """
    def __init__(self, env, valid_actions):
        super().__init__(env)
        self.valid_actions = valid_actions
        self.action_space = gym.spaces.Discrete(len(valid_actions))

class DQN(nn.Module):
    def __init__(self, in_channels=3, n_actions=3):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.view(-1, 3, 7, 7) 

        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))

        x = x.view(x.size(0), -1)  # flatten
        x = nn.functional.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

# Set up the environment
# env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human")
valid_actions = [Actions.left, Actions.right, Actions.forward]
env = gym.make(f"MiniGrid-Empty-{SIZE}x{SIZE}-v0", render_mode="rgb_array")
# env = FilterObservation(env, filter_keys=["direction", "image"])
env = EnvWrapper(env, valid_actions)
# env = FullyObsWrapper(env)
env = ImgObsWrapper(env)
# this wrapper both flattens the observation space, and one hot encodes the direction
# env = FlattenObservation(env)
obs_space = env.observation_space.shape
action_space = env.action_space.n
print(action_space)
print(obs_space)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_network = DQN(n_actions=action_space).to(device)
target_network = DQN(n_actions=action_space).to(device)
target_network.load_state_dict(q_network.state_dict())
# disables gradients for target_network
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Reward Tracking
rewards_per_episode = []

replay_buffer = []

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
                q_values = q_network(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device))
                action = q_values.argmax().item()
        
        # Take action in the environment
        next_obs, reward, done, truncated, info = env.step(action)
        extrinsic_reward = reward
        total_reward += 10* extrinsic_reward

        # Save in replay buffer
        replay_buffer.append((obs, action, total_reward, next_obs, done))
        if len(replay_buffer) > REPLAY_BUFFER:
            replay_buffer.pop(0)
        
        if len(replay_buffer) >= BATCH_SIZE:
            batch_sample = random.sample(replay_buffer, BATCH_SIZE)
            states, actions, total_rewards, next_states, dones = zip(*batch_sample)
            states = torch.tensor(states, dtype=torch.float32, device=device)
            actions = torch.tensor(actions, dtype=torch.int64, device=device)
            total_rewards = torch.tensor(total_rewards, dtype=torch.float32, device=device)
            next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
            dones = torch.tensor(dones, dtype=torch.float32, device=device)

            q_values = q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = target_network(next_states).max(1)[0].detach()
            target_q_values = total_rewards + DISCOUNT_FACTOR * next_q_values * (1-done)
        
            # Update Q-network
            loss = loss_fn(q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update episodic memory and state
            steps += 1
        
    # Update target network occasionally
    if episode % TARGET_NETWORK_UPDATES == 0:
        target_network.load_state_dict(q_network.state_dict())

    rewards_per_episode.append(total_reward)
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)  # Decay epsilon
    print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {EPSILON:.2f}, Reached target: {done}")

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
