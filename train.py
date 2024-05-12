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
BATCH_SIZE = 30                   # Batch size of training data
MINI_BATCH_SIZE = 50                # Number of episodes to take from the batch in precentage %
TOTAL_TIMESTEPS = NUM_STEPS * 7500  # 500   # Total timesteps to run
GAMMA = 0.99                        # Discount factor
GAE_LAM = 0.95                      # For generalized advantage estimation
NUM_EPOCHS = 500                    # Number of epochs to train
REPORT_STEPS = 1             # Number of timesteps between reports


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
        target_kl_div=0.2,
        max_policy_train_iters=10,
        value_train_iters=10,
        policy_lr=learning_rate,
        value_lr=learning_rate
    )

    ep_reward = 0.0
    ep_count = 0
    season_count = 0
    batch_count = 0

    pi_losses, v_losses, total_losses, approx_kls, stds, best_batch_reward = [], [], [], [], [], []
    max_rewards = []

    for t in range(NUM_EPOCHS):
        reward_vec, value_vec, log_vec, advantage_mean_vec, prev_obs_vec, obs_vec, action_vec = [[] for _ in
            range(BATCH_SIZE)], [[]for _ in range(BATCH_SIZE)], [[] for _ in range(BATCH_SIZE)], [[] for _ in range(BATCH_SIZE)], [[] for _ in range(BATCH_SIZE)], [[] for _ in
            range(BATCH_SIZE)], [[] for _ in range(BATCH_SIZE)]

        # reward_vec, value_vec, log_vec, advantage_mean_vec, prev_obs_vec, obs_vec, action_vec = [], [], [], [], [], [], []
        # for i in range(BATCH_SIZE):
        #     for item in zip(prev_obs_vec, obs_vec, action_vec, advantage_mean_vec, reward_vec, value_vec, log_vec):
        #         item.append([])
        if t % REPORT_STEPS == 0:
            print(t, '/', NUM_EPOCHS)
        best_ep_reward = 0
        for batch in range(BATCH_SIZE):
            obs, _ = env.reset()
            for step in range(NUM_STEPS):
                with torch.no_grad():
                    dist, values = trainer.network.forward(obs)
                    action, log_prob = trainer.network.sample(dist)
                    log_prob = torch.squeeze(log_prob).numpy()
                    values = torch.squeeze(values).numpy()

                clipped_action = np.clip(action, lower_bound, upper_bound)
                clipped_action = clipped_action.numpy()
                clipped_action = clipped_action[0]
                next_obs, reward, terminated, truncated, _ = env.step(clipped_action)
                done = terminated or truncated

                ep_reward += reward

                prev_obs_vec[batch].append(obs)
                obs_vec[batch].append(next_obs)
                action_vec[batch].append(action)
                reward_vec[batch].append(reward)
                log_vec[batch].append(log_prob)
                value_vec[batch].append(values)
                # Add to buffer

                # Calculate advantage and returns if it is the end of episode or
                # its time to update
                if done or step == NUM_STEPS:
                    if done:
                        ep_count += 1
                    if best_ep_reward < ep_reward:
                        best_ep_reward = ep_reward
                    break
        print("Batch complete")
        best_batch_reward.append(best_ep_reward)
        for i in range(BATCH_SIZE):
            season_count += 1
            #action_vec[i] = np.hstack(action_vec[i])
            #log_vec[i] = np.hstack(log_vec[i])
            #log_vec[i] = torch.tensor(log_vec[i])
            # Update for epochs
            gaes = policy.calculate_gaes(reward_vec[i], value_vec[i], obs_vec[i], trainer.network, GAMMA)
            value_vec[i] = np.hstack(value_vec[i])
            advantage = gaes - value_vec[i]
            advantage_mean = torch.mean(advantage)
            advantage_mean_vec[i].append(advantage_mean)
        #picking random observations
        train_data = [[] for _ in range(6)] #prev_obs_vec, obs_vec, action_vec, log_vec, advantage_mean_vec, reward_vec
        for mini_batch in range(int(BATCH_SIZE*100/MINI_BATCH_SIZE)):
            for batch in range(BATCH_SIZE):
                random_number = random.randint(0, len(obs_vec[batch])-1)
                train_data[0].append(prev_obs_vec[batch][random_number])
                train_data[1].append(obs_vec[batch][random_number])
                train_data[2].append(action_vec[batch][random_number])
                train_data[3].append(torch.tensor(log_vec[batch][random_number]))
                train_data[4].append(torch.tensor(advantage_mean_vec[batch]))
                train_data[5].append(reward_vec[batch][random_number])
        train_data[3] = torch.stack(train_data[3])
        train_data[4] = torch.stack(train_data[4])
        trainer.train_policy(train_data[0], train_data[1], train_data[2], train_data[3], train_data[4], train_data[5])

        ep_reward, ep_count = 0.0, 0

    # Save policy and value network
    Path('saved_network').mkdir(parents=True, exist_ok=True)
    trainer.network.save('saved_network/pi_network.pt', 'saved_network/v_network.pt')

    # Plot episodic reward
    # Create a figure and subplots
    fig, axs = plt.subplots(1, 4)
    x_values = range(len(trainer.value_loss))
    x2_values = range(len(trainer.kl_div_arr))
    trainer.actor_loss = np.hstack(trainer.actor_loss)
    trainer.policy_loss = np.hstack(trainer.policy_loss)
    trainer.value_loss = np.hstack(trainer.value_loss)
    axs[0].plot(x_values, trainer.value_loss)
    axs[0].set_title('value loss')
    axs[1].plot(x_values, trainer.actor_loss)
    axs[1].set_title('actor loss')
    axs[2].plot(x_values, trainer.policy_loss)
    axs[2].set_title('policy loss')
    axs[3].plot(x_values, trainer.kl_div_arr)
    axs[3].set_title('kl div stopped')
    #plt.tight_layout()
    plt.show()

    #play_new_game(policy)