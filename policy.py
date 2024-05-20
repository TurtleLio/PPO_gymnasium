
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
from collections import deque


device = 'cuda'


# Policy and value model
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_space_size, action_space_size, std=0.0):
        super().__init__()

        self.policy_layers = nn.Sequential(
            nn.Linear(obs_space_size, 1024),
            nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_size))

        self.value_layers = nn.Sequential(
            nn.Linear(obs_space_size, 1024),
            nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

        self.log_std = nn.Parameter(torch.ones(self.action_dim) *
                                    torch.log(torch.tensor(initial_std)),
                                    requires_grad=True)
        self.apply(init_weights)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def value(self, obs):
        obs_torch = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
        value = self.value_layers(obs_torch)
        return value

    def policy(self, obs):
        obs_torch = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
        policy_logits = self.policy_layers(obs_torch)
        std = self.log_std.exp()
        dist = Normal(policy_logits, std)
        return dist

    def forward(self, obs):
        obs_torch = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
        value = self.value_layers(obs_torch)
        mu = self.policy_layers(obs_torch)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        return dist, value

    def sample(self, dist):
        action = dist.sample()
        log_prob = torch.sum(dist.log_prob(action), dim=1)
        #log_prob = dist.log_prob(action)
        return action, log_prob

    def save(self, filepath_model_pi, filepath_model_v):
        """
        Save the model and optimizer parameters to the given filepath.
        """
        torch.save(self.policy_layers.state_dict(), filepath_model_pi)
        torch.save(self.value_layers.state_dict(), filepath_model_v)
        print("Saved PI network")
        return 1


class PPOTrainer(nn.Module):
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

        self.actor_loss = []
        self.value_loss = []
        self.policy_loss = []
        self.actor_loss_every_iter = []
        self.value_loss_every_iter = []
        self.policy_loss_every_iter = []
        self.kl_div_arr = []
        self.approx_kl = []
        self.std = []

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

        obs_torch = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0).to(device)

        dist, values = self.forward(obs_torch)

        action = dist.sample()

        log_prob = torch.sum(dist.log_prob(action), dim=1).to(device)

        return (action[0].detach().cpu().numpy(),
                torch.squeeze(log_prob).detach().cpu().numpy(),
                torch.squeeze(values).detach().cpu().numpy())

    def get_values(self, obs):
        """
        Function  to return value of the state
        """
        obs_torch = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0).to(device)

        _, values = self.forward(obs_torch)

        return torch.squeeze(values).detach().cpu().numpy()

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

        return (
            pi_loss.detach(),
            value_loss.detach(),
            total_loss.detach(),
            (torch.mean((ratio - 1) - torch.log(ratio))).detach(),
            torch.exp(self.log_std).detach()
        )
        actor_loss = torch.squeeze(pi_loss).detach().numpy()
        value_loss = torch.squeeze(value_loss).detach().numpy()
        policy_loss = torch.squeeze(total_loss).detach().numpy()
        self.actor_loss.append(actor_loss)
        self.value_loss.append(value_loss)
        self.policy_loss.append(policy_loss)

    def save(self, filepath_model_pi, filepath_model_v):
        """
        Save the model and optimizer parameters to the given filepath.
        """
        torch.save(self.pi_network.state_dict(), filepath_model_pi)
        torch.save(self.v_network.state_dict(), filepath_model_v)
        print("Saved PI network")
        return 1


    # def train_policy(self, obs, next_obs, acts, old_log_probs, advantage, rewards, old_values, returns):
    #     torch.autograd.set_detect_anomaly(True)
    #     rewards = torch.tensor(rewards)
    #     for i in range(self.max_policy_train_iters):
    #         new_log_probs_vec, values, entropys = [], [], []
    #         for ob in obs:
    #             dist, value = self.network.forward(ob)
    #             action, new_log_probs = self.network.sample(dist)
    #             entropy = dist.entropy().mean()
    #             new_log_probs_vec.append(new_log_probs)
    #             values.append(value)
    #             entropys.append(entropy)
    #         new_log_probs_vec = torch.stack(new_log_probs_vec)
    #         values = torch.stack(values)
    #         #values = torch.squeeze(values)
    #         entropys = torch.stack(entropys)
    #         #returns_stacked = torch.stack(returns)
    #         new_log_probs_vec = torch.squeeze(new_log_probs_vec)
    #         policy_ratio = torch.exp(new_log_probs_vec - old_log_probs)
    #         full_loss = policy_ratio * advantage
    #         #clipped_ratio = policy_ratio.clamp(
    #         #    1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
    #         clipped_ratio = torch.clamp(policy_ratio, 1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
    #         clipped_loss = clipped_ratio * advantage
    #         #actor_loss_not_mean = torch.where(full_loss < clipped_loss, full_loss, clipped_loss)
    #         #actor_loss = torch.mean(actor_loss_not_mean)
    #         actor_loss = torch.min(clipped_loss, full_loss).mean()
    #         # value_loss_not_mean = (rewards - values).pow(2)
    #         # value_loss = torch.mean(value_loss_not_mean)
    #         vf_loss1 = (returns - values).pow(2)
    #         values = torch.squeeze(values)
    #         #old_values = torch.tensor(old_values)
    #         vpredclipped = old_values + torch.clamp(values - old_values, - self.ppo_clip_val, self.ppo_clip_val)
    #         # Note that we ONLY backprop through the new value
    #         vf_loss2 = (vpredclipped - returns).pow(2)
    #
    #         # 3. Take the MAX between the two losses
    #         # This trick has the effect of only updating the current value DIRECTLY if is it WORSE (higher error)
    #         # than the old value.
    #         # If the old value was worse then the "approximation" will be worse and we update
    #         # the new value only a little bit!
    #         value_loss = torch.max(vf_loss1, vf_loss2).mean()
    #         entropys_mean = torch.mean(entropys)
    #         policy_loss = 0.5 * value_loss + actor_loss - 0.001 * entropys_mean
    #         if i == 0:
    #             print("start of training")
    #             print(f"value loss:{value_loss} | actor loss:{actor_loss} | policy loss:{policy_loss}")
    #         if i == self.max_policy_train_iters -1:
    #             print("end of training")
    #             print(f"value loss:{value_loss} | actor loss:{actor_loss} | policy loss:{policy_loss}")
    #         self.optim.zero_grad()
    #         policy_loss.backward()
    #         nn.utils.clip_grad_norm_(self.network.parameters(),40)
    #         self.optim.step()
    #         kl_div = (old_log_probs - new_log_probs).mean()
    #         if kl_div >= self.target_kl_div:
    #             print(f"target kl achived after {i} iterations")
    #             self.kl_div_arr.append(i)
    #             print("end of training")
    #             print(f"value loss:{value_loss} | actor loss:{actor_loss} | policy loss:{policy_loss}")
    #             break
    #     actor_loss = torch.squeeze(actor_loss).detach().numpy()
    #     value_loss = torch.squeeze(value_loss).detach().numpy()
    #     policy_loss = torch.squeeze(policy_loss).detach().numpy()
    #     self.actor_loss.append(actor_loss)
    #     self.value_loss.append(value_loss)
    #     self.policy_loss.append(policy_loss)

    # def mean_with_grad(self, tensor):
    #     """Calculates mean while accumulating gradients."""
    #     return tensor.sum() / len(tensor)

    # def train_value(self, obs, returns):
    #     for _ in range(self.value_train_iters):
    #         self.value_optim.zero_grad()
    #
    #         values = self.network.value(obs)
    #         value_loss = (returns - values) ** 2
    #         value_loss = value_loss.mean()
    #
    #         value_loss.backward(retain_graph=True)
    #         self.value_optim.step()


def compute_return_advantage(
        rewards, values, mask, gamma, gae_lambda, last_value):
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

    returns = advantages + values

    return returns, advantages


def get_mini_batch(train_data, batch_size):
    rng = np.random.default_rng(seed=42)
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
    temp_data = [[] for _ in range(8)]
    for i in range(8):
        # Gather the data for each index before splitting
        temp_data[i] = np.array([train_data[i][index] for index in indices])
    temp_data = [np.split(temp_data[i][indices], split_indices) for i in range(8)]

    # Prepare mini-batches as a list of dictionaries
    n = len(temp_data[0])
    data_out = []
    for k in range(n):
        data_out.append(
            {
                'prev_obs_vec': temp_data[0][k],
                'obs_vec': temp_data[1][k],
                'action_vec': temp_data[2][k],
                'log_vec': temp_data[3][k],
                'advantage': temp_data[4][k],
                'reward_vec': temp_data[5][k],
                'value_vec': temp_data[6][k],
                'returns': temp_data[7][k],
            }
        )
    # train_data[0] = torch.tensor(prev_obs_vec, dtype=torch.float32).to(device)
    # train_data[1] = torch.tensor(obs_vec, dtype=torch.float32).to(device)
    # train_data[2] = torch.tensor(action_vec, dtype=torch.float32).to(device)
    # train_data[3] = torch.tensor(log_vec, dtype=torch.float32).to(device)
    # train_data[4] = torch.tensor(advantage, dtype=torch.float32).to(device)
    # train_data[5] = torch.tensor(reward_vec, dtype=torch.float32).to(device)
    # train_data[6] = torch.tensor(value_vec, dtype=torch.float32).to(device)
    # train_data[7] = torch.tensor(returns, dtype=torch.float32).to(device)

    return data_out
# def discount_rewards(rewards, gamma=0.99):
#     """
#     Return discounted rewards based on the given rewards and gamma param.
#     """
#     new_rewards = [float(rewards[-1])]
#     for i in reversed(range(len(rewards)-1)):
#         new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
#     return np.array(new_rewards[::-1])
#
# def calculate_gaes(rewards, values, last_value, next_obs, masks, model, gamma=0.99, decay=0.95):
#     """
#     Return the General Advantage Estimates from the given rewards and values.
#     Paper: https://arxiv.org/pdf/1506.02438.pdf
#     """
#     # next_values = []
#     # for obs in next_obs:
#     #     with torch.no_grad():
#     #         next_value = model.value(obs)
#     #     next_values.append(next_value)
#     gae = 0
#     returns = deque()
#     #values = values + [last_value]
#     for step in reversed(range(len(rewards))):
#         delta = rewards[step] + gamma * last_value * masks[step] - values[step]
#         gae = delta + gamma * decay * masks[step] * gae
#         last_value = values[step]
#         # delta = rewards[step] + gamma * values[step + 1] - values[step]
#         # gae = delta + gamma * decay * gae
#         returns.appendleft(gae + values[step])
#     # deltas = [rew + gamma * next_value - val for rew, val, next_value in reversed(zip(rewards, values, next_values))]
#     # deltas_stacked = torch.FloatTensor(deltas)
#     # deltas_stacked = torch.stack(deltas_stacked)
#     return returns
#     #return deltas_stacked
#
# def rollout(model, env, max_steps=1000):
#     """
#     Performs a single rollout.
#     Returns training data in the shape (n_steps, observation_shape)
#     and the cumulative reward.
#     """
#     ### Create data storage
#     train_data = [[], [], [], [], []] # obs, act, reward, values, act_log_probs
#     obs = env.reset()
#
#     ep_reward = 0
#     for _ in range(max_steps):
#         logits, val = model(torch.tensor(obs[0], dtype=torch.float32,
#                                          device=DEVICE))
#         act_distribution = Categorical(logits=logits)
#         act = act_distribution.sample()
#         act_log_prob = act_distribution.log_prob(act).item()
#
#         act, val = act.item(), val.item()
#
#         next_obs, reward, done, terminated, info = env.step(act)
#
#         for i, item in enumerate((obs, act, reward, val, act_log_prob)):
#           train_data[i].append(item)
#
#         obs = next_obs
#         ep_reward = ep_reward + reward
#         if done:
#             break
#
#     train_data = [np.asarray(x) for x in train_data]
#
#     ### Do train data filtering
#     train_data[3] = calculate_gaes(train_data[2], train_data[3])
#
#     return train_data, ep_reward
