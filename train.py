#!/usr/bin/python3

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

import policy
#from policy import PPOPolicy


#from test_pendulum import play_new_game


NUM_STEPS = 5000                    # Timesteps data to collect before updating
BATCH_SIZE = 64                     # Batch size of training data
TOTAL_TIMESTEPS = NUM_STEPS * 1000  # 500   # Total timesteps to run
GAMMA = 0.99                        # Discount factor
GAE_LAM = 0.95                      # For generalized advantage estimation
NUM_EPOCHS = 500                    # Number of epochs to train
REPORT_STEPS = 1000                 # Number of timesteps between reports


class PI_Network(nn.Module):
    def __init__(self, obs_dim, action_dim, lower_bound, upper_bound) -> None:
        super().__init__()
        (
            self.lower_bound,
            self.upper_bound
        ) = (
            torch.tensor(lower_bound, dtype=torch.float32),
            torch.tensor(upper_bound, dtype=torch.float32)
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

        return action


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

def organizing_array(array):
    # Flatten the nested list and ndarray structure
    single_value_ndarrays = []
    for sublist in array:
        for item in sublist:
            single_value_ndarrays.append(np.array([item]))

    return single_value_ndarrays

if __name__ == "__main__":

    env = gym.make('BipedalWalker-v3')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    lower_bound = env.action_space.low
    upper_bound = env.action_space.high

    pi_network = PI_Network(obs_dim, action_dim, lower_bound, upper_bound)
    v_network = V_Network(obs_dim)

    learning_rate = 3e-4

    #buffer = PPOBuffer(obs_dim, action_dim, NUM_STEPS)

    trainer = policy.PPOPolicy(
        pi_network,
        v_network,
        learning_rate,
        clip_range=0.2,
        value_coeff=0.5,
        obs_dim=obs_dim,
        action_dim=action_dim,
        initial_std=1.0,
        max_grad_norm=0.5,
    )

    ep_reward = 0.0
    ep_count = 0
    season_count = 0

    pi_losses, v_losses, total_losses, approx_kls, stds = [], [], [], [], []
    mean_rewards = []

    obs, _ = env.reset()
    reward_vec, value_vec, log_vec, advantage_vec, prev_obs_vec, obs_vec, action_vec, returns_vec, masks = [], [], [], [], [], [], [], [], []
    train_data = [[] for _ in range(8)]  # prev_obs_vec, obs_vec, action_vec, log_vec, advantage_mean_vec, reward_vec, value_vec, return_vec
    last_value_vec = 0
    best_episode_reward = 0
    best_episode_reward_vec = []
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
        # Add to buffer
        #buffer.record(obs, action, reward, values, log_prob)

        obs = next_obs

        # Calculate advantage and returns if it is the end of episode or
        # its time to update
        if done or (t + 1) % NUM_STEPS == 0:
            if done:
                ep_count += 1
            # Value of last time-step
            if best_episode_reward < ep_reward or best_episode_reward == 0:
                best_episode_reward = ep_reward
            ep_reward = 0
            last_value = trainer.get_values(obs)

            # Compute returns and advantage and store in buffer
            # buffer.process_trajectory(
            #     gamma=GAMMA,
            #     gae_lam=GAE_LAM,
            #     is_last_terminal=done,
            #     last_v=last_value)
            returns, advantage = policy.compute_return_advantage(reward_vec, value_vec, masks, GAMMA, GAE_LAM,
                                                             last_value)
            returns_vec.append(returns)
            advantage_vec.append(advantage)
            ep_reward = 0
            obs, _ = env.reset()

        if (t + 1) % NUM_STEPS == 0:

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
            print(f"best episode reward = {best_episode_reward}")
            best_episode_reward_vec.append(best_episode_reward)
            best_episode_reward = 0
            season_count += 1
            for i in range(8):
                train_data[i] = np.concatenate(train_data[i])
            # Update for epochs
            for ep in range(NUM_EPOCHS):
                batch_data = policy.get_mini_batch(train_data, BATCH_SIZE)
                num_grads = len(batch_data)

                # Iterate over minibatch of data
                for k in range(num_grads):
                    (
                        obs_batch,
                        action_batch,
                        log_prob_batch,
                        advantage_batch,
                        return_batch,
                    ) = (
                        batch_data[k]['prev_obs_vec'],
                        batch_data[k]['action_vec'],
                        batch_data[k]['log_vec'],
                        batch_data[k]['advantage'],
                        batch_data[k]['returns'],
                    )

                    # Normalize advantage
                    advantage_batch = (
                        advantage_batch -
                        np.squeeze(np.mean(advantage_batch, axis=0))
                    ) / (np.squeeze(np.std(advantage_batch, axis=0)) + 1e-8)

                    # Convert to torch tensor
                    (
                        obs_batch,
                        action_batch,
                        log_prob_batch,
                        advantage_batch,
                        return_batch,
                    ) = (
                        torch.tensor(obs_batch, dtype=torch.float32),
                        torch.tensor(action_batch, dtype=torch.float32),
                        torch.tensor(log_prob_batch, dtype=torch.float32),
                        torch.tensor(advantage_batch, dtype=torch.float32),
                        torch.tensor(return_batch, dtype=torch.float32),
                    )

                    # Update the networks on minibatch of data
                    (
                        pi_loss,
                        v_loss,
                        total_loss,
                        approx_kl,
                        std, stop
                    ) = trainer.update(obs_batch, action_batch,
                                      log_prob_batch, advantage_batch,
                                      return_batch)
                    if ep == 0 and k == 0:
                        pi_losses.append(pi_loss.numpy())
                        v_losses.append(v_loss.numpy())
                        total_losses.append(total_loss.numpy())
                        approx_kls.append(approx_kl.numpy())
                        stds.append(std.numpy())
                        print(f"start of training: pi loss: {pi_loss} | v loss: {v_loss}| total loss: {total_loss}")
                if stop:
                    print(f"target kl achived after {ep} iterations")
                    break
            print(f"end of training: pi loss: {pi_loss} | v loss: {v_loss}| total loss: {total_loss}")
            #buffer.clear()
            reward_vec, value_vec, log_vec, advantage_vec, prev_obs_vec, obs_vec, action_vec, returns_vec, masks = [], [], [], [], [], [], [], [], []
            train_data = [[] for _ in range(8)]  # prev_obs_vec, obs_vec, action_vec, log_vec, advantage_mean_vec, reward_vec, value_vec, return_vec
            mean_ep_reward = ep_reward / ep_count
            ep_reward, ep_count = 0.0, 0
            last_value_vec = 0
            best_ep_reward = 0


            mean_rewards.append(mean_ep_reward)
            pi_losses, v_losses, total_losses, approx_kls, stds = (
                    [], [], [], [], [])

    # Save policy and value network
    Path('saved_network_new_game').mkdir(parents=True, exist_ok=True)
    torch.save(pi_network.state_dict(), 'saved_network_new_game/pi_network.pth')
    torch.save(v_network.state_dict(), 'saved_network_new_game/v_network.pth')

    # Plot episodic reward
    _, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
    ax.plot(range(season_count), mean_rewards)
    ax.set_xlabel("season")
    ax.set_ylabel("episodic reward")
    ax.grid(True)
    plt.show()

    plt.figure()
    plt.plot(range(len(best_episode_reward_vec)), best_episode_reward_vec)
    plt.title('best episode reward')
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.show()

    plt.figure()
    plt.plot(range(len(pi_losses)), pi_losses)
    plt.title('pi_losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    plt.plot(range(len(v_losses)), v_losses)
    plt.title('v_losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    plt.plot(range(len(total_losses)), total_losses)
    plt.title('total_losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    #play_new_game(policy)