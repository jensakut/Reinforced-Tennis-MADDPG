from collections import namedtuple

import numpy as np
import torch

from .SumTree import SumTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# based on https://github.com/rlcode/per

class PrioritizedExperienceReplay:
    """Fixed-size buffer to store experience tuples."""
    """
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    """

    def __init__(self, args):
        """Initialize a ExperienceReplay object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        # self.seed = args.random_seed
        self.max_priority = args.per_max_priority
        self.alpha = args.per_alpha
        self.beta = args.per_beta
        self.alpha_increment = (args.per_alpha_end - args.per_alpha) / args.per_annihilation
        self.beta_increment = (args.per_beta_end - args.per_beta) / args.per_annihilation
        self.eps = args.per_eps
        self.i_episode = 0  # set from agent
        self.max_priority = args.per_max_priority
        self.tree = SumTree(args.buffer_size)
        self.capacity = args.buffer_size
        self.batch_size = args.batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["obs", "obs_full", "actions", "actions_full", "rewards", "next_obs",
                                                  "next_obs_full", "dones"])

    def _get_priority(self, error):
        # Sample transition probability P(i) = p_i^alpha / sum_k (p_k^alpha)
        # instead, just safe priorities with the alpha squared already.
        return (np.abs(error) + self.eps) ** self.alpha

    def add(self, obs, actions, rewards, next_obs, dones):
        """Add a new experience to memory."""
        # assume that these incoming items are lists of each agent's observations
        obs_full = np.hstack(obs)
        actions_full = np.hstack(actions)
        next_obs_full = np.hstack(next_obs)

        experience = self.experience(obs, obs_full, actions, actions_full, rewards, next_obs, next_obs_full, dones)

        p = self.max_priority
        self.tree.add(p, experience)

        # done signals episode has ended. Then change alpha and beta
        # alpha and beta outside 0...1 makes no sense
        if np.any(dones):
            self.alpha = np.max(np.min([1., self.alpha + self.alpha_increment]), 0)
            self.beta = np.max(np.min([1., self.beta + self.beta_increment]), 0)

    def sample(self):
        """Prioritized randomly sample a batch of experiences from memory."""
        experiences = []
        idxs = []
        priorities = []

        # sample a batch of random experiences by drawing a float between 0 and the sum of the tree
        s = np.random.uniform(0, self.tree.total(), self.batch_size)
        for i in range(self.batch_size):
            (idx, p, experience) = self.tree.get(s[i])
            priorities.append(p)
            experiences.append(experience)
            idxs.append(idx)

        # pseudocode 10: compute importance-sampling weight w_j = (N + P(j))^-beta / max_i w_i
        sampling_probabilities = priorities / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

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

        is_weights = torch.from_numpy(is_weights).float().to(device)
        return idxs, is_weights, obs, obs_full, actions, actions_full, rewards, next_obs, next_obs_full, dones

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = self._get_priority(error)
            self.tree.update(idx, p)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.tree.n_entries
