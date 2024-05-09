
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical
from torch.distributions import Normal


DEVICE = 'cpu'


# Policy and value model
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_space_size, action_space_size, std=0.0):
        super().__init__()

        self.policy_layers = nn.Sequential(
            nn.Linear(obs_space_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_size))

        self.value_layers = nn.Sequential(
            nn.Linear(obs_space_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

        self.log_std = nn.Parameter(torch.ones(1, action_space_size) * std)
        self.apply(init_weights)

    def value(self, obs):
        value = self.value_layers(obs)
        return value

    def policy(self, obs):
        policy_logits = self.policy_layers(obs)
        return policy_logits

    def forward(self, obs):
        #observation = obs.clone()
        obs_torch = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
        value = self.value_layers(obs_torch)
        mu = self.policy_layers(obs_torch)
        #std = torch.exp(self.log_std)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        value = torch.squeeze(value).detach().numpy()
        return dist, value

    def sample(self, dist):
        action = dist.sample()
        #log_prob = dist.log_prob(action).sum(dim=1)
        log_prob = torch.sum(dist.log_prob(action), dim =1)
        log_prob = torch.squeeze(log_prob).detach().numpy()
        return action, log_prob

    def save(self, filepath_model_pi, filepath_model_v, filepath_model_shared):
        """
        Save the model and optimizer parameters to the given filepath.
        """
        #torch.save(self.shared_layers.state_dict(), filepath_model_shared)
        torch.save(self.policy_layers.state_dict(), filepath_model_pi)
        torch.save(self.value_layers.state_dict(), filepath_model_v)
        print("Saved PI network")
        return 1


class PPOTrainer():
    def __init__(self,
                 actor_critic,
                 ppo_clip_val=0.2,
                 target_kl_div=0.01,
                 max_policy_train_iters=10,
                 value_train_iters=10,
                 policy_lr=1e-4,
                 value_lr=1e-4):
        self.network = actor_critic
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters

        self.actor_loss = []
        self.value_loss = []
        self.policy_loss = []
        self.actor_loss_every_iter = []
        self.value_loss_every_iter = []
        self.policy_loss_every_iter = []

        self.optim = optim.Adam(self.network.parameters(), lr=value_lr)


    def train_policy(self, obs, next_obs, acts, old_log_probs, gaes, rewards):
        torch.autograd.set_detect_anomaly(True)
        for i in range(self.max_policy_train_iters):
            dist, value = self.network(obs)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(acts)
            policy_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = policy_ratio.clamp(
                1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
            clipped_loss = clipped_ratio * gaes
            full_loss = policy_ratio * gaes
            actor_loss_not_mean = torch.where(full_loss < clipped_loss, full_loss, clipped_loss)
            actor_loss = torch.mean(actor_loss_not_mean)
            value_loss_not_mean = (rewards - value).pow(2)
            value_loss = torch.mean(value_loss_not_mean)
            policy_loss = value_loss + actor_loss - 0.01 * entropy
            if i == 0:
                print("start of training")
                print(f"value loss:{value_loss} | actor loss:{actor_loss} | policy loss:{policy_loss}")
            if i == self.max_policy_train_iters -1:
                print("end of training")
                print(f"value loss:{value_loss} | actor loss:{actor_loss} | policy loss:{policy_loss}")
            # self.actor_loss_every_iter.append(actor_loss)
            # self.value_loss_every_iter.append(value_loss)
            # self.policy_loss_every_iter.append(policy_loss)
            self.optim.zero_grad()
            policy_loss.backward()
            self.optim.step()
            kl_div = (old_log_probs - new_log_probs).mean()
            if kl_div >= self.target_kl_div:
                print(f"target kl achived after {i} iterations")
                break
        self.actor_loss.append(actor_loss)
        self.value_loss.append(value_loss)
        self.policy_loss.append(policy_loss)

    def mean_with_grad(self, tensor):
        """Calculates mean while accumulating gradients."""
        return tensor.sum() / len(tensor)

    def train_value(self, obs, returns):
        for _ in range(self.value_train_iters):
            self.value_optim.zero_grad()

            values = self.network.value(obs)
            value_loss = (returns - values) ** 2
            value_loss = value_loss.mean()

            value_loss.backward(retain_graph=True)
            self.value_optim.step()

def discount_rewards(rewards, gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_gaes(rewards, prev_values, next_obs, model, gamma=0.99, decay=0.95):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = []
    for obs in next_obs:
        with torch.no_grad():
            next_value = model.value(obs)
        next_values.append(next_value)
    deltas = [rew + gamma * next_value - val for rew, val, next_value in zip(rewards, prev_values, next_values)]
    deltas_stacked = torch.stack(deltas)
    return deltas_stacked

def rollout(model, env, max_steps=1000):
    """
    Performs a single rollout.
    Returns training data in the shape (n_steps, observation_shape)
    and the cumulative reward.
    """
    ### Create data storage
    train_data = [[], [], [], [], []] # obs, act, reward, values, act_log_probs
    obs = env.reset()

    ep_reward = 0
    for _ in range(max_steps):
        logits, val = model(torch.tensor(obs[0], dtype=torch.float32,
                                         device=DEVICE))
        act_distribution = Categorical(logits=logits)
        act = act_distribution.sample()
        act_log_prob = act_distribution.log_prob(act).item()

        act, val = act.item(), val.item()

        next_obs, reward, done, terminated, info = env.step(act)

        for i, item in enumerate((obs, act, reward, val, act_log_prob)):
          train_data[i].append(item)

        obs = next_obs
        ep_reward = ep_reward + reward
        if done:
            break

    train_data = [np.asarray(x) for x in train_data]

    ### Do train data filtering
    train_data[3] = calculate_gaes(train_data[2], train_data[3])

    return train_data, ep_reward
