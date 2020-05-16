import copy
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from maddpg.ddpg_model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(local_model, target_model, tau):
    """
    soft update model parameters
    :param local_model: PyTorch model (weights will be copied from)
    :param target_model: PyTorch model (weights will be copied to)
    :param tau: interpolation parameter
    :return:
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class DDPGAgent:
    def __init__(self, state_size_local, state_size_full, action_size, action_size_full, par):
        # Actor Network (w/ Target Network)
        # the Actor network receives only the local observation
        self.actor_local = Actor(state_size_local, action_size, par).to(device)
        self.actor_target = Actor(state_size_local, action_size, par).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=par.lr_actor)
        print('actor')
        print(self.actor_local)

        # Critic Network (w/ Target Network)
        # for maddpg the critic receives the full observation of all agents
        self.critic_local = Critic(state_size_full, action_size_full, par).to(device)
        self.critic_target = Critic(state_size_full, action_size_full, par).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=par.lr_critic,
                                           weight_decay=par.weight_decay)
        print('critic')
        print(self.critic_local)
        # initialize targets same as original networks

        hard_update(self.actor_local, self.actor_target)
        hard_update(self.critic_local, self.critic_target)

        # Noise process
        self.noise = OUNoise(action_size, par.ou_mu, par.ou_theta, par.ou_sigma)
        self.memory = ReplayBuffer(par.buffer_size, par.batch_size)
        self.par = par

    def act(self, state, epsilon, add_noise=True):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(device)
        # set actor in evaluation mode
        self.actor_local.eval()
        # do not memorize gradients
        with torch.no_grad():
            action = self.actor_local.eval()(state).cpu().data.numpy()
        # set actor back to training mode
        self.actor_local.train()

        if add_noise:
            action += epsilon * self.noise.sample()
        # clipping is done with tanh output layers
        np.clip(action, -1, 1)
        return action


class MADDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, par, num_agents, num_instances=1):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state from the agents perspective for the actors
            state_size_full (int): the state size of the full observation for the critics
            action_size (int): dimension of each action
            random_seed (int): random seed
            par (Par): parameter object
        """
        # self.state_size = state_size
        # self.state_size_full = state_size_full

        self.num_instances = num_instances

        self.action_size = action_size
        # self.seed = random.seed(par.random_seed)

        self.ddpg_agents = []
        for _ in range(num_agents):
            self.ddpg_agents.append(DDPGAgent(state_size, state_size * num_agents, action_size,
                                              action_size * num_agents, par))

        self.time_learn = deque(maxlen=100)
        self.time_act = deque(maxlen=100)
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
        # todo change to allow for more than the number of agents in a game
        # save individual memory to agent's individual replay buffer
        for i, ddpg_agent in enumerate(self.ddpg_agents):
            # agent's local observation
            obs = states[i]
            next_obs = next_states[i]
            # full observation for critic
            obs_full = states.flatten()
            next_obs_full = next_states.flatten()
            actions_full = np.asarray(actions).flatten()
            ddpg_agent.memory.add(obs, obs_full, np.asarray(actions[i]), actions_full, np.asarray(rewards[i]), next_obs,
                                  next_obs_full, np.asarray(dones[i]))

        # Learn, if enough samples are available in memory
        if len(self.ddpg_agents[i].memory) > self.par.batch_size * 2 and timestep % self.par.update_every == 0:
            self.learn(self.ddpg_agents)

        if np.any(dones):
            # try a different noise-driven trajectory every episode
            self.reset()
            self.epsilon *= self.par.epsilon_decay

    def act(self, states, add_noise=True):
        actions = []
        for state, ddpg_agent in zip(states, self.ddpg_agents):
            # expand state to be a batch of one for batch normalization
            state_expanded = np.expand_dims(state, axis=0)
            actions.append(ddpg_agent.act(state_expanded, self.epsilon, add_noise))
        return actions

    def reset(self):
        for ddpg_agent in self.ddpg_agents:
            ddpg_agent.noise.reset()

    def learn(self, ddpg_agents):
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
        # train each agent independently
        for iii, ddpg_agent in enumerate(self.ddpg_agents):
            # sample some experiences and unpack
            experiences = ddpg_agent.memory.sample()
            obs, obs_full, actions, actions_full, rewards, next_obs, next_obs_full, dones = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models from the perspective of the agent
            # compute the action of each target actor for the critic
            actions_nexts = []
            for ii, agent in enumerate(ddpg_agents):
                actions_nexts.append(agent.actor_target(next_obs).detach().numpy())
            actions_next_full = torch.from_numpy(np.hstack(actions_nexts)).float().to(device)

            # the critic works with the entire observation flattened
            Q_targets_next = ddpg_agent.critic_target(next_obs_full, actions_next_full)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.par.gamma * Q_targets_next * (1 - dones))

            # Compute critic loss
            Q_expected = ddpg_agent.critic_local(obs_full, actions_full)
            critic_loss = F.mse_loss(Q_expected, Q_targets)

            # Minimize the loss
            ddpg_agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpg_agent.critic_local.parameters(), 1)
            ddpg_agent.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            # compute the action of all agents to compute the value with the critic
            actions_preds = []
            for ii, agent in enumerate(self.ddpg_agents):
                actions_preds.append(agent.actor_local(obs).detach().numpy())
            actions_pred_full = torch.from_numpy(np.hstack(actions_preds)).float().to(device)
            actor_loss = -ddpg_agent.critic_local(obs_full, actions_pred_full).mean()

            # Minimize the loss
            ddpg_agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            ddpg_agent.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            soft_update(ddpg_agent.critic_local, ddpg_agent.critic_target, self.par.tau)
            soft_update(ddpg_agent.actor_local, ddpg_agent.actor_target, self.par.tau)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.05, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array(
            [random.random() for i in range(len(self.state))])
        self.state += dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """
        Initialize a replay buffer object
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["obs", "obs_full", "actions", "actions_full", "rewards", "next_obs",
                                                  "next_obs_full", "dones"])

    def add(self, obs, obs_full, actions, actions_full, rewards, next_obs, next_obs_full, dones):
        """ Add a new experience to memory"""
        e = self.experience(obs, obs_full, actions, actions_full, rewards, next_obs, next_obs_full, dones)
        self.memory.append(e)

    def sample(self):
        """ Randomly sample a batch of experiences from memory and return it on device """
        experiences = random.sample(self.memory, k=self.batch_size)
        obs = torch.from_numpy(np.vstack([e.obs for e in experiences if e is not None])).float().to(device)
        obs_full = \
            torch.from_numpy(np.vstack([e.obs_full for e in experiences if e is not None])).float().to(device)
        actions = \
            torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float().to(device)
        actions_full = \
            torch.from_numpy(np.vstack([e.actions_full for e in experiences if e is not None])).float().to(device)
        rewards = \
            torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_obs = \
            torch.from_numpy(np.vstack([e.next_obs for e in experiences if e is not None])).float().to(device)
        next_obs_full = \
            torch.from_numpy(np.vstack([e.next_obs_full for e in experiences if e is not None])).float().to(device)
        dones = \
            torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(
                device)
        # for maddpg these are a list of paired experiences
        return obs, obs_full, actions, actions_full, rewards, next_obs, next_obs_full, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
