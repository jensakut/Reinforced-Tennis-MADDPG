class Par:
    def __init__(self):
        # Parameters suggested according to this paper: Continuous control with deep reinforcement learning
        # https://arxiv.org/pdf/1509.02971.pdf

        # Learning hyperparameters
        self.buffer_size = int(1e6)  # replay buffer size
        self.batch_size = 64  # minibatch size
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters
        self.lr_actor = 1e-4  # learning rate of the actor
        self.lr_critic = 1e-3  # learning rate of the critic
        self.weight_decay = 1e-2  # L2 weight decay

        # ou noise
        self.ou_mu = 0.
        self.ou_theta = 0.15
        self.ou_sigma = 0.25

        # network architecture for actor and critic
        self.actor_fc1_units = 400
        self.actor_fc2_units = 300
        self.critic_fcs1_units = 400
        self.critic_fc2_units = 300

        # Further parameter not found in paper
        self.random_seed = 15  # random seed
        self.update_every = 16  # time steps between updates
        self.num_updates = 1  # num of update passes when updating
        self.epsilon = 1.0  # epsilon for the noise process added to the actions
        self.epsilon_decay = 1  # decay for epsilon above
        self.num_episodes = 1000  # number of episodes
        self.file_name = 'Tennis_Linux/Tennis.x86_64'
        self.file_name_watch = self.file_name
        self.train = True


class ParTennis(Par):
    def __init__(self):
        super(ParTennis, self).__init__()

        # tuned parameter to "reach" the goal
        # Learning
        self.batch_size = 512  # mini batch size
        self.lr_actor = 1e-4  # learning rate of the actor
        self.lr_critic = 1e-4  # learning rate of the critic
        self.tau = 1e-2

        # ou noise
        self.ou_theta = 0.5
        self.ou_sigma = 0.2

        # network architecture for actor and critic
        self.actor_fc1_units = 96
        self.actor_fc2_units = 64
        self.critic_fcs1_units = 96
        self.critic_fc2_units = 64

        self.update_every = 16  # time steps between updates
        # self.num_updates = 16  # num of update passes when updating

        self.epsilon_decay = 0.9999  # 1e-6  # decay for epsilon above

        self.num_episodes = 2500  # number of episodes
