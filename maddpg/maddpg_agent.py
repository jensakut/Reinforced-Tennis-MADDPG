import copy
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from maddpg.PrioritizedExperienceReplay import PrioritizedExperienceReplay
from maddpg.maddpg_model import Actor, Critic

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
        # print('actor')
        # print(self.actor_local)

        # Critic Network (w/ Target Network)
        # for maddpg the critic receives the full observation of all agents
        self.critic_local = Critic(state_size_full, action_size_full, par).to(device)
        self.critic_target = Critic(state_size_full, action_size_full, par).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=par.lr_critic,
                                           weight_decay=par.weight_decay)

        self.mse_loss = nn.MSELoss()
        self.mse_element_loss = nn.MSELoss()

        # print('critic')
        # print(self.critic_local)
        # initialize targets same as original networks

        hard_update(self.actor_local, self.actor_target)
        hard_update(self.critic_local, self.critic_target)

        # Noise process
        self.noise = OUNoise(action_size, par.ou_mu, par.ou_theta, par.ou_sigma, par.random_seed)
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
        # unpack action which has an unneccessary []
        return action[0]

    def save_model(self, experiment_name, i_episode):
        torch.save(self.actor_local.state_dict(), experiment_name + '_checkpoint_actor_' + str(i_episode) + '.pth')
        torch.save(self.critic_local.state_dict(), experiment_name + '_checkpoint_critic_' + str(i_episode) + '.pth')


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
        self.seed = random.seed(par.random_seed)

        self.ddpg_agents = []
        for _ in range(num_agents):
            self.ddpg_agents.append(DDPGAgent(state_size, state_size * num_agents, action_size,
                                              action_size * num_agents, par))

        # self.time_learn = deque(maxlen=100)
        # self.time_act = deque(maxlen=100)
        self.epsilon = par.epsilon
        self.par = par
        # Replay memory
        if par.use_prioritized_experience_replay:
            self.memory = PrioritizedExperienceReplay(par)
        else:
            self.memory = ReplayBuffer(par.buffer_size, par.batch_size)

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
        self.memory.add(states, np.asarray(actions), np.asarray(rewards), next_states, np.asarray(dones))

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.par.batch_size * 4 and timestep % self.par.update_every == 0:
            self.learn()

        if np.any(dones):
            # try a different noise-driven trajectory every episode
            self.reset()
            self.epsilon *= self.par.epsilon_decay

        # if random.randint(0, 10) is 0:
        #     self.reset()

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

    def learn(self):
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

        # Train the agent using the experience out of each perspective

        # sample some experiences and unpack
        experiences = self.memory.sample()
        if self.par.use_prioritized_experience_replay:
            sample_indices, is_weights, obs, obs_full, actions, actions_full, rewards, next_obs, next_obs_full, dones = experiences
        else:
            obs, obs_full, actions, actions_full, rewards, next_obs, next_obs_full, dones = experiences

        # get empty list of priorities to average later on
        priorities = []

        # Get predicted next-state actions and Q values from target models from the perspective of the agent
        # compute the action of each target actor for the critic
        actions_nexts = []
        for it1, agent in enumerate(self.ddpg_agents):
            next_obs_agent_view = next_obs[:, it1, :]
            actions_nexts.append(agent.actor_target(next_obs_agent_view).detach().cpu().numpy())
        actions_next_full = torch.from_numpy(np.hstack(actions_nexts)).float().to(device)

        # to train the actor prepare output for critic: compute the action of all agents using the current policy
        actions_preds = []
        for it2, agent in enumerate(self.ddpg_agents):
            obs_agent_view = obs[:, it2, :]
            actions_preds.append(agent.actor_local(obs_agent_view).detach().cpu().numpy())
        actions_pred_full = torch.from_numpy(np.hstack(actions_preds)).float().to(device)

        # train each agent independently
        for it, ddpg_agent in enumerate(self.ddpg_agents):
            # ---------------------------- update critic ---------------------------- #
            # the critic works with the entire observation flattened
            Q_targets_next = ddpg_agent.critic_target(next_obs_full, actions_next_full)

            # Compute Q targets for current states (y_i) for current agent 'it'

            Q_targets = rewards[:, it].reshape(-1, 1) + (self.par.gamma * Q_targets_next *
                                                         (1 - dones[:, it].reshape(-1, 1)))
            # Compute critic loss
            Q_expected = ddpg_agent.critic_local(obs_full, actions_full)
            if self.par.use_prioritized_experience_replay:
                # pseudocode 11, loss is td_error and pseudocode 12 priorities is abs td_error
                priorities.append(abs(Q_targets - Q_expected))
                # pseudocode 13
                critic_loss = (is_weights * ddpg_agent.mse_element_loss(Q_expected, Q_targets)).mean()
                # Update Priorities based on offseted TD error
            else:
                # loss is the difference between currently estimated value and value provided by the tuples and the target
                # network
                critic_loss = ddpg_agent.mse_loss(Q_expected, Q_targets)

            # Minimize the loss
            ddpg_agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpg_agent.critic_local.parameters(), 1)
            ddpg_agent.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            # take actions from all actors
            actor_loss = -ddpg_agent.critic_local(obs_full, actions_pred_full).mean()

            # Minimize the loss
            ddpg_agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            ddpg_agent.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            soft_update(ddpg_agent.critic_local, ddpg_agent.critic_target, self.par.tau)
            soft_update(ddpg_agent.actor_local, ddpg_agent.actor_target, self.par.tau)

        if self.par.use_prioritized_experience_replay:
            priorities_numpy = []
            for priority in priorities:
                priorities_numpy.append(priority.squeeze().to('cpu').data.numpy())

            priorities_mean = np.mean(priorities_numpy, axis=0)
            self.memory.update_priorities(sample_indices, priorities_mean)

    def save_model(self, experiment_name, i_episode):
        for agent in self.ddpg_agents:
            agent.save_model(experiment_name, i_episode)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.05, sigma=0.2, seed=48, ):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        self.seed = random.seed(seed)

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
        self.experience = namedtuple("Experience", field_names=["obs", "obs_full", "actions", "actions_full", "rewards",
                                                                "next_obs", "next_obs_full", "dones"])

    def add(self, obs, actions, rewards, next_obs, dones):
        """ Add a new experience to memory"""
        # assume that these incoming items are lists of each agent's observations
        obs_full = np.hstack(obs)
        actions_full = np.hstack(actions)
        next_obs_full = np.hstack(next_obs)

        e = self.experience(obs, obs_full, actions, actions_full, rewards, next_obs, next_obs_full, dones)
        self.memory.append(e)

    def sample(self):
        """ Randomly sample a batch of experiences from memory and return it on device """
        experiences = random.sample(self.memory, k=self.batch_size)

        obs = torch.from_numpy(np.asarray([e.obs for e in experiences if e is not None])).float().to(device)
        obs_full = \
            torch.from_numpy(np.asarray([e.obs_full for e in experiences if e is not None])).float().to(device)
        actions = \
            torch.from_numpy(np.asarray([e.actions for e in experiences if e is not None])).float().to(device)
        actions_full = \
            torch.from_numpy(np.asarray([e.actions_full for e in experiences if e is not None])).float().to(device)
        rewards = \
            torch.from_numpy(np.asarray([e.rewards for e in experiences if e is not None])).float().to(device)
        next_obs = \
            torch.from_numpy(np.asarray([e.next_obs for e in experiences if e is not None])).float().to(device)
        next_obs_full = \
            torch.from_numpy(np.asarray([e.next_obs_full for e in experiences if e is not None])).float().to(device)
        dones = \
            torch.from_numpy(np.asarray([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(
                device)
        # for maddpg these are a list of paired experiences
        return obs, obs_full, actions, actions_full, rewards, next_obs, next_obs_full, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
