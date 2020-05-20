import copy
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg.ddpg_model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent():
    def __init__(self, state_size, action_size, par):
        self.par = par
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, par).to(device)
        self.actor_target = Actor(state_size, action_size, par).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=par.lr_actor)
        print('actor')
        print(self.actor_local)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, par).to(device)
        self.critic_target = Critic(state_size, action_size, par).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=par.lr_critic,
                                           weight_decay=par.weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, par.random_seed, par.ou_mu, par.ou_theta, par.ou_sigma)

    def save_model(self, experiment_name, i_episode):
        path = self.par.save_path
        torch.save(self.actor_local.state_dict(), experiment_name + '_checkpoint_actor_' + str(i_episode) + '.pth')
        torch.save(self.critic_local.state_dict(), experiment_name + '_checkpoint_critic_' + str(i_episode) + '.pth')


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, par, num_agents=1):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            par (Par): parameter object
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(par.random_seed)
        self.epsilon = par.epsilon

        self.ddpg_agents = []
        for _ in range(num_agents):
            self.ddpg_agents.append(DDPGAgent(state_size, action_size, par))

        # Replay memory
        self.memory = ReplayBuffer(action_size, par.buffer_size, par.batch_size, par.random_seed)

        # Make sure target is with the same weight as the source
        # The seed makes sure the networks are the same
        # self.hard_copy(self.actor_target, self.actor_local)
        # self.hard_copy(self.critic_target, self.critic_local)

        self.time_learn = deque(maxlen=100)
        self.time_act = deque(maxlen=100)
        self.actor_losses = []
        self.critic_losses = []
        self.learn_count = []
        self.epsilon = 1

        self.par = par

    def step(self, states, actions, rewards, next_states, dones, timestep):
        """
        Save experience in replay memory and use random sample from buffer to learn.
        :param states: state of the environment
        :param actions: executed action
        :param rewards: observed reward
        :param next_states: subsequent state
        :param dones: boolean signal indicating a finished episode
        :return:
        """

        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.par.batch_size * 2 and timestep % self.par.update_every == 0:
            self.learn(self.par.gamma)
        if np.any(dones):
            # ---------------------------- update noise ---------------------------- #
            self.epsilon *= self.par.epsilon_decay
            self.reset()

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""

        actions = []
        """
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        """

        for i, ddpg_agent in enumerate(self.ddpg_agents):
            state = torch.from_numpy(np.asarray([states[i]])).float().to(device)

            ddpg_agent.actor_local.eval()
            with torch.no_grad():
                actions.append(ddpg_agent.actor_local(state).cpu().data.numpy())
            ddpg_agent.actor_local.train()

            if add_noise:
                actions[-1] += self.epsilon * ddpg_agent.noise.sample()
        # clipping is done with tanh output layers
        return actions

    def reset(self):
        for ddpg_agent in self.ddpg_agents:
            ddpg_agent.noise.reset()

    def learn(self, gamma):
        for ddpg_agent in self.ddpg_agents:
            experiences = self.memory.sample()
            actor_loss, critic_loss = self._learn(experiences, gamma, ddpg_agent)
            self.actor_losses.append(actor_loss)
            self.critic_losses.append(critic_loss)

    def _learn(self, experiences, gamma, ddpg_agent):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = ddpg_agent.actor_target(next_states)
        Q_targets_next = ddpg_agent.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = ddpg_agent.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        ddpg_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(ddpg_agent.critic_local.parameters(), 1)
        ddpg_agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = ddpg_agent.actor_local(states)
        actor_loss = -ddpg_agent.critic_local(states, actions_pred).mean()

        # Minimize the loss
        ddpg_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        ddpg_agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(ddpg_agent.critic_local, ddpg_agent.critic_target, self.par.tau)
        self.soft_update(ddpg_agent.actor_local, ddpg_agent.actor_target, self.par.tau)
        return 0, 0

    def soft_update(self, local_model, target_model, tau):
        """
        soft update model parameters
        :param local_model: PyTorch model (weights will be copied from)
        :param target_model: PyTorch model (weights will be copied to)
        :param tau: interpolation parameter
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_copy(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save_model(self, experiment_name, i_episode):
        for agent in self.ddpg_agents:
            agent.save_model(experiment_name, i_episode)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.05, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize a replay buffer object
        :param action_size:
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param seed:
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """ Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """ Randomly sample a batch of experiences from memory and return it on device """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
