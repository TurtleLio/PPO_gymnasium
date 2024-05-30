import torch
from torch import nn
from torch.distributions import Normal
import numpy as np


class PPOPolicy(nn.Module):
    def __init__(self, pi_network, v_network, learning_rate, clip_range,
                 value_coeff, obs_dim, action_dim, initial_std=1.0,
                 max_grad_norm=0.5):

        super().__init__()

        (
            self.pi_network,
            self.v_network,
            self.learning_rate,
            self.clip_range,
            self.value_coeff,
            self.obs_dim,
            self.action_dim,
            self.max_grad_norm,
        ) = (
            pi_network,
            v_network,
            learning_rate,
            clip_range,
            value_coeff,
            obs_dim,
            action_dim,
            max_grad_norm
        )

        # Gaussian policy will be used. So, log standard deviation is created
        # as trainable variables
        self.log_std = nn.Parameter(torch.ones(self.action_dim) *
                                    torch.log(torch.tensor(initial_std)),
                                    requires_grad=True)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.learning_rate)

    def forward(self, obs):
        pi_out = self.pi_network(obs)

        # Add Normal distribution layer at the output of pi_network
        dist_out = Normal(pi_out, torch.exp(self.log_std))

        v_out = self.v_network(obs)

        return dist_out, v_out

    def get_action(self, obs):
        """
        Sample action based on current policy
        """

        obs_torch = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)

        dist, values = self.forward(obs_torch)

        action = dist.sample()

        log_prob = torch.sum(dist.log_prob(action), dim=1)

        return (action[0].detach().numpy(),
                torch.squeeze(log_prob).detach().numpy(),
                torch.squeeze(values).detach().numpy())

    def get_values(self, obs):
        """
        Function  to return value of the state
        """
        obs_torch = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)

        _, values = self.forward(obs_torch)

        return torch.squeeze(values).detach().numpy()

    def evaluate_action(self, obs_batch, action_batch, training):
        """
        Evaluate taken action.
        """
        obs_torch = obs_batch.clone().detach()
        action_torch = action_batch.clone().detach()
        dist, values = self.forward(obs_torch)
        log_prob = dist.log_prob(action_torch)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return log_prob, values

    def update(self, obs_batch, action_batch, log_prob_batch, advantage_batch,
               return_batch):
        """
        Performs one step gradient update of policy and value network.
        """

        new_log_prob, values = self.evaluate_action(
                obs_batch, action_batch, training=True)

        ratio = torch.exp(new_log_prob-log_prob_batch)
        clipped_ratio = torch.clip(
            ratio,
            1-self.clip_range,
            1+self.clip_range,
        )

        surr1 = ratio * advantage_batch
        surr2 = clipped_ratio * advantage_batch
        pi_loss = -torch.mean(torch.min(surr1, surr2))
        value_loss = self.value_coeff * torch.mean((values - return_batch)**2)
        total_loss = pi_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        stop = 0
        kl_div = (log_prob_batch - new_log_prob).mean()
        #old_log = log_prob_batch.mean()
        if kl_div >= 0.1:
            stop = 1

        return (
            pi_loss.detach(),
            value_loss.detach(),
            total_loss.detach(),
            (torch.mean((ratio - 1) - torch.log(ratio))).detach(),
            torch.exp(self.log_std).detach(),
            stop
        )

def compute_return_advantage(rewards, values, mask, gamma, gae_lambda, last_value):
    """
    Computes returns and advantage based on generalized advantage estimation.
    """
    #N = rewards.shape[0]
    N = len(rewards)
    advantages = np.zeros(
        (N, 1),
        dtype=np.float32
    )

    tmp = 0.0
    for k in reversed(range(N)):
        if k == (N-1):
            next_values = last_value
        # if k == N - 1:
        #     next_non_terminal = 1 - is_last_terminal
        #     next_values = last_value
        else:
            #next_non_terminal = 1
            next_values = values[k+1]
        next_non_terminal = mask[k]
        delta = (rewards[k] +
                 gamma * next_non_terminal * next_values -
                 values[k])

        tmp = delta + gamma * gae_lambda * next_non_terminal * tmp

        advantages[k] = tmp
    advantages = [item for sublist in advantages for item in sublist]
    values = np.array(values)
    returns = advantages + values

    return returns, advantages


def get_mini_batch(train_data, batch_size):
    rng = np.random.default_rng()
    # Check that batch_size is smaller than the number of data points
    assert batch_size <= len(train_data[0]), \
        "Batch size must be smaller than number of data."

    # Get the number of data points
    data_size = len(train_data[0])

    # Create and shuffle indices
    indices = np.arange(data_size)
    rng.shuffle(indices)

    # Split indices into mini-batches
    split_indices = []
    point = batch_size
    while point < data_size:
        split_indices.append(point)
        point += batch_size

    # Split data based on shuffled indices
    temp_data = {
        'prev_obs_vec': np.split(train_data[0][indices], split_indices),
        'obs_vec': np.split(train_data[1][indices], split_indices),
        'action_vec': np.split(train_data[2][indices], split_indices),
        'log_vec': np.split(train_data[3][indices], split_indices),
        'advantage': np.split(train_data[4][indices], split_indices),
        'reward_vec': np.split(train_data[5][indices], split_indices),
        'value_vec': np.split(train_data[6][indices], split_indices),
        'returns': np.split(train_data[7][indices], split_indices)
    }
    # temp_data = [[] for _ in range(8)]
    # for i in range(8):
    #     # Gather the data for each index before splitting
    #     temp_data[i] = np.array([train_data[i][index] for index in indices])
    # temp_data = [np.split(temp_data[i][indices], split_indices) for i in range(8)]
    # Prepare mini-batches as a list of dictionaries
    n = len(temp_data['prev_obs_vec'])
    data_out = []
    for k in range(n):
        data_out.append(
            {
                'prev_obs_vec': temp_data['prev_obs_vec'][k],
                'obs_vec': temp_data['obs_vec'][k],
                'action_vec': temp_data['action_vec'][k],
                'log_vec': temp_data['log_vec'][k],
                'advantage': temp_data['advantage'][k],
                'reward_vec': temp_data['reward_vec'][k],
                'value_vec': temp_data['value_vec'][k],
                'returns': temp_data['returns'][k],
            }
        )

    return data_out