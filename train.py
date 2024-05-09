#!/usr/bin/python3

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

import policy
import random

#from test_pendulum import play_new_game


NUM_STEPS = 5000                    # Timesteps data to collect before updating
BATCH_SIZE = 20                   # Batch size of training data
TOTAL_TIMESTEPS = NUM_STEPS * 50  # 500   # Total timesteps to run
GAMMA = 0.99                        # Discount factor
GAE_LAM = 0.95                      # For generalized advantage estimation
NUM_EPOCHS = 500                    # Number of epochs to train
REPORT_STEPS = 1000               # Number of timesteps between reports
random_reset = random.randint(1, 5000)

if __name__ == "__main__":

    env = gym.make('BipedalWalker-v3')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    lower_bound = env.action_space.low
    upper_bound = env.action_space.high

    network = policy.ActorCriticNetwork(obs_dim, action_dim)

    learning_rate = 3e-4

    trainer = policy.PPOTrainer(
        network,
        ppo_clip_val=0.2,
        target_kl_div=0.01,
        max_policy_train_iters=10,
        value_train_iters=10,
        policy_lr=learning_rate,
        value_lr=learning_rate
    )

    ep_reward = 0.0
    ep_count = 0
    season_count = 0
    batch_count = 0

    pi_losses, v_losses, total_losses, approx_kls, stds = [], [], [], [], []
    reward_vec, value_vec, log_vec, gae_lambda_vec = [], [], [], []
    prev_obs_vec, obs_vec, action_vec = [], [], []
    max_rewards = []

    obs, _ = env.reset()

    for t in range(TOTAL_TIMESTEPS):

        if t % REPORT_STEPS == 0:
            print(t, '/', TOTAL_TIMESTEPS)
        with torch.no_grad():
            dist, values = trainer.network.forward(obs)
            action, log_prob = trainer.network.sample(dist)
            log_prob = torch.squeeze(log_prob).numpy()
            values = torch.squeeze(values).numpy()

        clipped_action = np.clip(action, lower_bound, upper_bound)
        clipped_action = clipped_action.numpy()
        clipped_action = clipped_action[0]
        next_obs, reward, terminated, truncated, _ = env.step(clipped_action)
        if (t+1) % random_reset == 0:
            terminated = True
        done = terminated or truncated

        ep_reward += reward

        # Add to buffer
        #buffer.record(obs, action, reward, values, log_prob)

        # Calculate advantage and returns if it is the end of episode or
        # its time to update
        if done or (t + 1) % NUM_STEPS == 0:
            if done:
                ep_count += 1
            batch_count += 1
            # buffer
            prev_obs_vec.append(obs)
            obs_vec.append(next_obs)
            action_vec.append(action)
            reward_vec.append(reward)
            log_vec.append(log_prob)
            value_vec.append(values)
            random_reset = random.randint(1, 5000)
            obs, _ = env.reset()
            next_obs = obs
        obs = next_obs

        if (batch_count+1) % BATCH_SIZE == 0:
            prev_obs_vec.append(obs)
            obs_vec.append(next_obs)
            action_vec.append(action)
            reward_vec.append(reward)
            log_vec.append(log_prob)
            value_vec.append(values)
            season_count += 1
            action_vec = torch.stack(action_vec)
            log_vec = np.hstack(log_vec)
            log_vec = torch.tensor(log_vec)
            # Update for epochs
            gaes = policy.calculate_gaes(reward_vec, value_vec, obs_vec, trainer.network, GAMMA)
            value_vec = np.hstack(value_vec)
            advantage = gaes - value_vec
            advantage_mean = torch.mean(advantage)
            trainer.train_policy(prev_obs_vec, obs_vec, action_vec, log_vec, advantage_mean, reward_vec)
            max_reward = np.max(np.array(reward_vec))

            ep_reward, ep_count = 0.0, 0
            batch_count = 0
            prev_obs_vec, obs_vec, action_vec, reward_vec, log_vec, value_vec, max_rewards= [], [], [], [], [], [], []

            random_reset = random.randint(1, 5000)
            obs, _ = env.reset()
            next_obs = obs

    # Save policy and value network
    Path('saved_network').mkdir(parents=True, exist_ok=True)
    trainer.network.save('saved_network/pi_network.pt', 'saved_network/v_network.pt')

    # Plot episodic reward
    # Create a figure and subplots
    fig, axs = plt.subplots(1, 3)
    x_values = range(len(trainer.value_loss))
    trainer.actor_loss = np.hstack(trainer.actor_loss)
    trainer.policy_loss = np.hstack(trainer.policy_loss)
    trainer.value_loss = np.hstack(trainer.value_loss)
    axs[0].plot(x_values, trainer.value_loss)
    axs[0].set_title('value_loss')
    axs[1].plot(x_values, trainer.actor_loss)
    axs[1].set_title('actor_loss')
    axs[2].plot(x_values, trainer.policy_loss)
    axs[2].set_title('policy_loss')
    plt.tight_layout()
    plt.show()

    #play_new_game(policy)