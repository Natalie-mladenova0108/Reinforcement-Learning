import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, RGBImgObsWrapper
from minigrid.core.actions import Actions
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class EnvWrapper(gym.Wrapper):
    """
    For now, this just minimizes the action space of the environment.
    (It is set to 7, although 4 of them are unused)
    """
    def __init__(self, env, valid_actions):
        super().__init__(env)
        self.valid_actions = valid_actions
        self.action_space = gym.spaces.Discrete(len(valid_actions))


# SB3 DQN kernel's default size is 8x8, which is too big for Minigrid
# We require a custom kernel: https://minigrid.farama.org/content/training/ 
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

valid_actions = [Actions.left, Actions.right, Actions.forward]
env = gym.make("MiniGrid-Empty-16x16-v0")
env = EnvWrapper(env, valid_actions)
env = RGBImgObsWrapper(env)
env = ImgObsWrapper(env) 
# env = gym.wrappers.TransformObservation(env, lambda obs: obs.transpose(2, 0, 1), None)
print(env.observation_space)
# env.observation_space = gym.spaces.Box(0, 255, (3, 64, 64), np.uint8)
print(env.observation_space)

def train_model():
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = DQN(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=100,
        batch_size=32,
        gamma=0.99,
        target_update_interval=100,
        train_freq=4,
        # idk about gradient_steps
        gradient_steps=1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        exploration_fraction=0.2,
        verbose=1,
    )

    model.learn(
        total_timesteps=5000,
        progress_bar=True,
    )

    model.save("dqn_minigrid_16")

def evaluate_model():
    model = DQN.load("dqn_minigrid_16")

    MAX_STEPS = 300
    obs = env.reset()
    num_episodes = 100

    for _ in range(num_episodes):
        done = False
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        while steps < MAX_STEPS and not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            # only extrinsic reward for now
            total_reward += reward
        print(f"Reward received: {total_reward}; Reached target: {done}")

train_model()
evaluate_model()

# Results for 5x5: consistently 0.955
# Results for 8x8: consistently 0.961328125
# Results for 16x16:

env.close()