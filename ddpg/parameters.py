class Par:
    def __init__(self):
        # Parameters suggested according to this paper: Continuous control with deep reinforcement learning
        # https://arxiv.org/pdf/1509.02971.pdf

        # Learning hyperparameters
        self.buffer_size = int(1e6)  # replay buffer size
        self.batch_size = 256  # minibatch size
        self.gamma = 0.9  # discount factor
        self.tau = 1e-3  # for soft update of target parameters
        self.lr_actor = 1e-3  # learning rate of the actor
        self.lr_critic = 1e-3  # learning rate of the critic
        self.weight_decay = 0  # L2 weight decay

        # ou noise
        self.ou_mu = 0.
        self.ou_theta = 0.15
        self.ou_sigma = 0.01

        # network architecture for actor and critic
        self.actor_fc1_units = 32
        self.actor_fc2_units = 16
        self.critic_fcs1_units = 32
        self.critic_fc2_units = 16

        # Further parameter not found in paper
        self.random_seed = 15  # random seed
        self.update_every = 1  # timesteps between updates
        self.epsilon = 1.0  # epsilon for the noise process added to the actions
        self.epsilon_decay = 1  # 1e-6  # decay for epsilon above

        self.num_episodes = 5000  # number of episodes
        self.file_name = 'Tennis_Linux/Tennis.x86_64'
        self.file_name_watch = 'Tennis_Linux/Tennis.x86_64'
        self.save_path = '../results/ddpg/'
