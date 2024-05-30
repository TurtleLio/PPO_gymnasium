from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

import policy
from train import PI_Network, V_Network


def play_new_game(network):
    env_play = gym.make('BipedalWalker-v3', render_mode="human")
    epochs, reward, reward_sum =0, 0, 0
    done, truncated, terminated = False, False, False

    state, info = env_play.reset()
    while not done:
        action, log_prob, values = network.get_action(state)

        clipped_action = np.clip(action, env_play.action_space.low, env_play.action_space.high)
        # clipped_action = clipped_action.numpy()
        # clipped_action = clipped_action[0]
        state, reward, terminated, truncated, _ = env_play.step(clipped_action)

        reward_sum += reward
        done = terminated or truncated
        epochs += 1
        if epochs >= 500:
            done = True
    print(f"reward: {reward_sum}")

    env_play.reset()
    env_play.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('BipedalWalker-v3')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    lower_bound = env.action_space.low
    upper_bound = env.action_space.high
    pi_network = PI_Network(obs_dim,action_dim,lower_bound,upper_bound).to(device)
    v_network = V_Network(obs_dim).to(device)
    pi_network.load_state_dict(torch.load('saved_network/pi_network.pt'))
    v_network.load_state_dict(torch.load('saved_network/v_network.pt'))

    trainer = policy.PPOTrainer(
        pi_network,
        v_network,
        learning_rate=0.004,
        clip_range=0.2,
        value_coeff=0.5,
        obs_dim=obs_dim,
        action_dim=action_dim,
        initial_std=1.0,
        max_grad_norm=0.5,
    ).to(device)

    play_new_game(trainer)