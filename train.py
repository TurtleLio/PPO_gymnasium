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

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
class PI_Network(nn.Module):
    def __init__(self, obs_dim, action_dim, lower_bound, upper_bound) -> None:
        super().__init__()
        (
            self.lower_bound,
            self.upper_bound
        ) = (
            torch.tensor(lower_bound, dtype=torch.float32).to(device),
            torch.tensor(upper_bound, dtype=torch.float32).to(device)
        )
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, obs):
        y = F.tanh(self.fc1(obs))
        y = F.tanh(self.fc2(y))
        action = self.fc3(y)

        action = ((action + 1) * (self.upper_bound - self.lower_bound) / 2 +
                  self.lower_bound)
        nan_mask = torch.isnan(action)
        action_cleaned = torch.where(nan_mask, torch.tensor(0.0), action)

        return action_cleaned


class V_Network(nn.Module):
    def __init__(self, obs_dim) -> None:
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs):
        y = F.tanh(self.fc1(obs))
        y = F.tanh(self.fc2(y))
        values = self.fc3(y)

        return values


def decide_epochs(best_reward):
    if best_reward > -100:
        return 500
    elif best_reward > -110:
        return 250
    else:
        return 100


def organizing_array(array):
    # Flatten the nested list and ndarray structure
    single_value_ndarrays = []
    for sublist in array:
        for item in sublist:
            single_value_ndarrays.append(np.array([item]))

    return single_value_ndarrays

#worked kinda good: batch_size=10, mini_batch_size=20, num_epochs=500
NUM_STEPS = 5000                   # Timesteps data to collect before updating
BATCH_SIZE = 64                # Batch size of training data
#MINI_BATCH_SIZE = 64          # Number of episodes to take from the batch in precentage %
TOTAL_TIMESTEPS = NUM_STEPS * 1000  # 500   # Total timesteps to run
GAMMA = 0.99                        # Discount factor
GAE_LAM = 0.95                      # For generalized advantage estimation
NUM_EPOCHS = 500                   # Number of epochs to train
#NUM_EPISODES = 1000
REPORT_STEPS = 1000             # Number of timesteps between reports


if __name__ == "__main__":
    env = gym.make('BipedalWalker-v3')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    lower_bound = env.action_space.low
    upper_bound = env.action_space.high

    pi_network = PI_Network(obs_dim, action_dim, lower_bound, upper_bound).to(device)
    v_network = V_Network(obs_dim).to(device)

    learning_rate = 3e-4

    trainer = policy.PPOTrainer(
        pi_network,
        v_network,
        learning_rate,
        clip_range=0.2,
        value_coeff=0.5,
        obs_dim=obs_dim,
        action_dim=action_dim,
        initial_std=1.0,
        max_grad_norm=0.5
    ).to(device)

    ep_reward = 0
    ep_count = 0
    season_count = 0
    batch_count = 0

    pi_losses, v_losses, total_losses, approx_kls, stds, best_batch_reward = [], [], [], [], [], []
    max_rewards = []
    obs, _ = env.reset()
    reward_vec, value_vec, log_vec, advantage_vec, prev_obs_vec, obs_vec, action_vec, returns_vec, masks = [], [], [], [], [], [], [], [], []
    train_data = [[] for _ in range(8)]  # prev_obs_vec, obs_vec, action_vec, log_vec, advantage_mean_vec, reward_vec, value_vec, return_vec
    last_value_vec = 0
    best_ep_reward = 0
    for t in range(TOTAL_TIMESTEPS):
        if t % REPORT_STEPS == 0:
            print(t, '/', TOTAL_TIMESTEPS)
        action, log_prob, values = trainer.get_action(obs)
        clipped_action = np.clip(action, lower_bound, upper_bound)
        next_obs, reward, terminated, truncated, _ = env.step(clipped_action)
        done = terminated or truncated

        ep_reward += reward
        values = values.item()
        log_prob = log_prob.item()
        masks.append(1 - done)
        prev_obs_vec.append(obs)
        obs_vec.append(next_obs)
        action_vec.append(action)
        reward_vec.append(reward)
        log_vec.append(log_prob)
        value_vec.append(values)

        obs = next_obs
                # Calculate advantage and returns if it is the end of episode or
                # its time to update
        if done or (t+1) % NUM_STEPS == 0:
            if done:
                ep_count += 1
            if best_ep_reward < ep_reward or best_ep_reward == 0:
                best_ep_reward = ep_reward
            last_value = trainer.get_values(obs)
            returns, advantage = policy.compute_return_advantage(reward_vec, value_vec, masks, GAMMA, GAE_LAM,
                                                             last_value)
            returns_vec.append(returns)
            advantage_vec.append(advantage)
            ep_reward = 0
            obs, _ = env.reset()
        if (t+1) % NUM_STEPS == 0:
            advantage_vec = organizing_array(advantage_vec)
            returns_vec = organizing_array(returns_vec)
            train_data[0].append(prev_obs_vec)
            train_data[1].append(obs_vec)
            train_data[2].append(action_vec)
            train_data[3].append(log_vec)
            train_data[4].append(advantage_vec)
            train_data[5].append(reward_vec)
            train_data[6].append(value_vec)
            train_data[7].append(returns_vec)
            print("Batch complete")
            print(f"best reward: {best_ep_reward}")
            best_batch_reward.append(best_ep_reward)
            season_count += 1

            for i in range(8):
                train_data[i] = np.concatenate(train_data[i])

            for i in range(NUM_EPOCHS):
                train_data_randomized = policy.get_mini_batch(train_data, BATCH_SIZE)
                for k in range(len(train_data_randomized)):
                    train_data_randomized[k]['advantage'] = (train_data_randomized[k]['advantage'] - np.squeeze(np.mean(train_data_randomized[k]['advantage'], axis=0))) / (np.squeeze(np.std(train_data_randomized[k]['advantage'], axis=0)) + 1e-8)
                    (
                        train_data_randomized[k]['prev_obs_vec'], train_data_randomized[k]['action_vec'], train_data_randomized[k]['log_vec'], train_data_randomized[k]['advantage'],
                        train_data_randomized[k]['returns']
                    ) = (
                        torch.tensor(train_data_randomized[k]['prev_obs_vec'], dtype=torch.float32).to(device),
                        torch.tensor(train_data_randomized[k]['action_vec'], dtype=torch.float32).to(device),
                        torch.tensor(train_data_randomized[k]['log_vec'], dtype=torch.float32).to(device),
                        torch.tensor(train_data_randomized[k]['advantage'], dtype=torch.float32).to(device),
                        torch.tensor(train_data_randomized[k]['returns'], dtype=torch.float32).to(device),
                    )
                    pi_loss,v_loss,total_loss,approx_kl,std, stop = trainer.update(train_data_randomized[k]['prev_obs_vec'], train_data_randomized[k]['action_vec'],train_data_randomized[k]['log_vec'], train_data_randomized[k]['advantage'],train_data_randomized[k]['returns'])
                if stop:
                    print(f"target kl achived after {i} iterations")
                    if stop:
                        break
                trainer.actor_loss.append(pi_loss.cpu().numpy())
                trainer.value_loss.append(v_loss.cpu().numpy())
                trainer.policy_loss.append(total_loss.cpu().numpy())
                trainer.approx_kl.append(approx_kl.cpu().numpy())
                trainer.std.append(std.cpu().numpy())

                if i == 0:
                    print(f"Start of epochs: actor loss: {pi_loss} | value loss: {v_loss} | policy loss: {total_loss}")
                if i % 100 == 0:
                    print(f"epoch : {i}/{NUM_EPOCHS}")
            print(f"End of epochs: actor loss: {pi_loss} | value loss: {v_loss} | policy loss: {total_loss}")
            ep_reward, ep_count = 0, 0
            reward_vec, value_vec, log_vec, advantage_vec, prev_obs_vec, obs_vec, action_vec, returns_vec, masks = [], [], [], [], [], [], [], [], []
            train_data = [[] for _ in range(8)]  # prev_obs_vec, obs_vec, action_vec, log_vec, advantage_mean_vec, reward_vec, value_vec, return_vec
            last_value_vec = 0
            best_ep_reward = 0
    # Save policy and value network
    Path('saved_network').mkdir(parents=True, exist_ok=True)
    trainer.save('saved_network/pi_network.pt', 'saved_network/v_network.pt')

    # Plot episodic reward
    # Create a figure and subplots
    # fig, axs = plt.subplots(1, 5)
    x_values = range(len(trainer.value_loss))
    x2_values = range(len(trainer.kl_div_arr))
    x3_values = range(len(best_batch_reward))
    trainer.actor_loss = np.hstack(trainer.actor_loss)
    trainer.policy_loss = np.hstack(trainer.policy_loss)
    trainer.value_loss = np.hstack(trainer.value_loss)
    # axs[0].plot(x_values, trainer.value_loss)
    # axs[0].set_title('value loss')
    # axs[1].plot(x_values, trainer.actor_loss)
    # axs[1].set_title('actor loss')
    # axs[2].plot(x_values, trainer.policy_loss)
    # axs[2].set_title('policy loss')
    # axs[3].plot(x2_values, trainer.kl_div_arr)
    # axs[3].set_title('kl div stopped')
    # axs[4].plot(x3_values, best_batch_reward)
    # axs[4].set_title('kl div stopped')
    plt.figure()
    plt.plot(x_values, trainer.value_loss)
    plt.title('value loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.figure()
    plt.plot(x_values, trainer.actor_loss)
    plt.title('actor loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.figure()
    plt.plot(x_values, trainer.policy_loss)
    plt.title('policy loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.figure()
    plt.plot(x2_values, trainer.kl_div_arr)
    plt.title('kl div stopped')
    plt.xlabel('epoch')
    plt.ylabel('iteration')

    plt.figure()
    plt.plot(x3_values, best_batch_reward)
    plt.title('best reward')
    plt.xlabel('epoch')
    plt.ylabel('reward')
    #plt.tight_layout()
    plt.show()

    #play_new_game(policy)