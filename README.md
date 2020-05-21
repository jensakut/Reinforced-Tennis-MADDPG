#### Udacity Deep Reinforcement Learning Nanodegree 
## Project 2: Competitive Environments
# A Reinforced Tennis Match with DDPG and MADDPG Agents 
### Introduction


[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.



### Getting Started and training the best configuration

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    - Extract the environment into the project folder. 

1. Create (and activate) a new [anaconda](https://www.anaconda.com/distribution/) environment with Python 3.6.

Anaconda takes care of the right cuda installation, as long as a nvidia gpu with the official driver is installed. 
Therfore an installation is well worth it if not already done. Installing cuda manually is ... much more time-consuming 
and difficult.


    - __Linux__ or __Mac__: 
    ```bash
    conda env create -f environment.yml
    conda activate tennis 
    ```
    - __Windows__: 
    ```bash
    conda create --f environment.yml
    activate tennis
    ```
2. Install the requirements with 
    ```bash
    pip install .
    ```

3. Train the agent or watch a smart agent with the following scripts
    ```bash
    python ddpg/train.py
    python ddpg/watch.py
    ```
   or 
    ```bash
    python maddpg/train.py
    python maddpg/watch.py
    ```
   
# Background
## Policy-based methods 

### Definition of policy-based vs. value-based: 

Value-based methods use experienced state-action-reward-state tuples with the environment to estimate the optimal action-value 
function. The optimal policy is derived by choosing a policy which maximizes the expected value. 
Policy-based methods on the other hand directly learn the optimal policy, without maintaining the value estimate. 

Deep reinforcement learning represents the policy within a neural network. The input is the observed environment state. 
For a discrete output, the layer has a node for each possible action which shows the execution probability. 
At first, the network is initialized randomly. Then, the agent learns a policy as it interacts with the environment. 

Policy-based methods can learn either stochastic or deterministic policies, and the action space can either be finite 
or continous.

## Policy-based Methods for reinforced learning

### Hill Climbing

Hill climbing iteratively finds the weights θ for an optimal policy. 
For each iteration: 
- The values are slightly changed to yield a new set of weights
- A new episode is collected. If the new weights yield a higher reward, these weights are set
as the current best estimate. 

### Improving Hill Climbing 

- Steepest ascent hill climbing choses a small number of neighbouring policies at each iteration and chooses 
the best among them. This helps finding the best way towards the optimum. 
- Simulated annealing uses a pre-defined radius to control how the policy space is explored, which is reduced while closing in 
on the optimal solution. This makes search more efficient. 
- Adaptive noise scaling decreases the search radius with each improvement of a policy, while increasing the radius if no improvement was found. 
This makes it likely that a policy doesn't get stuck in a local optimum.    

### Methods beyond hill climbing: 
- The cross-entropy method iterates over neighbouring policies and uses the best performing policies to calculate a new estimate. 
- The evolution strategies method considers the return of each candidate policy. A reward-weighted sum over all candidate policies 
uses all available information. 

### Why policy-based methods? 
- Simplicity: Policy-based methods directly estimate the policy without the intermitting value function. This is more efficient 
in particular with respect to a vast action-space in which an estimate of each action has to be kept.  
- Stochastic policies: Policy-based methods learn true stochastic policies 
- Continous action spaces: Policy-based methods are well-suited for continous action spaces withouth the need for discretization. 

## Policy-gradient methods

In contrast to policy-based methods, policy-gradient methods use the gradient of the policy. They are a subclass of 
policy-based methods. The policy is nowadays often a neural network, in which the gradient is used to search for the 
optimal weights. 

The policy gradient method will use trajectories of state-action-reward-nextstate-nextaction to make actions with 
higher expected reward more likely.

A trajectory tau is a state-action sequence. The goal can be further clarified as the gradient is used to maximise the
expected reward of a given (set of) trajectories, because it is inefficient or impossible to compute the real gradient of all
possible trajectories. 

### Reinforce

The commonly used first policy-gradient method is reinforce. It uses the policy to collect a set of m trajectories with 
a horizon H. These are used to estimate the gradient and thus update the weights of the policy-neural network. This is 
repeated until a satisfactory score is achieved. 

This algorithm can solve MDPs with either stochastic or deterministic, discrete or continous action spaces. The latter was difficult with value-based
methods because the action needs to be discretized. DQNs can only use stochastic actions by leveraging the exploration factor, which is neither efficient nor pretty. 

### Proximal policy optimization (PPO)

Reinforce can be optimized using the following elements: Noise Reduction, Rewards Normalization, Credit Assignment, 
Importance Sampling with re-weighting to come up with the PPO algorithm. 
So that’s it! We can finally summarize the PPO algorithm

    - Collect some trajectories using the current policy π_θ
    - Initialize theta prime 
        θ′=θ
    - Compute the gradient of the clipped surrogate function using the trajectories
    - Update θ′ using gradient ascent 
        θ′←θ′+α∇θ′L_min_clipped_surrogate(θ′,θ)
    - Then we repeat step 2-3 without generating new trajectories. Typically, step 2-3 are only repeated a few times
    - Set theta θ=θ′, go back to step 1, repeat.


## Actor-critic methods

There are many different actor critic methods. Typically, the actor is a network computing an action, which the critic 
evaluates with assuming the state-value, state-action-value, or advantage of the gained state transition. 
The advantage is the difference in value of the states plus the gained reward. 
The actor policy-based network is then trained using the value the critic assumes. 
Therefore, the actor is a policy method and the critic is a value method. The critic helps to reduce the variance of 
policy-gradient method which have a high variance. 

The value-based critic can use a td-method or a monte-carlo method to estimate the value. A vanilla td-estimate means, 
that a single step, containing state, action, reward, next_state, next_action tuples are used to compute a gradient
to train the network. 
A monte-carlo estimate samples one trajectory of the game, then uses the reward to train the value-based network. 
Monte Carlo methods wait until the real reward of a trajectory is known, therefore they are unbiased. Since a trajectory
consists of many steps which in sum make up the rewards, a lot of variance is in the estimate. Imagine using entire
chess games to evaluate the value of one step. There are so many influences, that the variance is high. 
A TD-method has less variance, but is biased because an estimate is used to learn the estimate. 
A good method combines the least amount of variance with that amount of bias that is still capable of learning the
 optimal function. 
One compromise is an n-step td-method called n-step bootstrapping, in which n steps are used to estimate the value. Typically 5-6 steps are good,
but it varies across the problems to be solved. 
[Generalized advantage estimation (GAE)](https://arxiv.org/abs/1506.02438) can use a parameter to interpolate between td-estimate and monte-carlo-estimate
using a weighing factor. Depending on this weight, the future steps will be decayed, therefore a gradient is built 
using a mixture of the two.

Examples for actor-critic networks are: 
- A3C: Asynchronous Advantage Actor-Critic, N-step Bootstrapping
- A3C: Asynchronous Advantage Actor-Critic, Parallel Training
- A2C: Advantage Actor-Critic
- DDPG: Deep Deterministic Policy Gradient, Continuous Action-space

## DDPG

DDPG is an untypical actor-critic method and it could be seen as approximate dqn. The reason for this is that the critic
 in the DDPG is used to approximate the maximizer over the q-values of the next state and not as a learned baseline as
  in typical actor-critic methods. But this is still a very important algorithm. 
  
A limitation of the dqn agent is that it is not straightforward to use in continous deterministic action space. Discretizing
a continous action-space suffers from the curse of dimensionality and does not scale well. 
In DDPG two neural networks are used. One is the actor policy which computes an action based on both the state and the 
neural weights. The critic computes a state-value based on the neural weights. 
 The actor is learning the argmax Q(s,a) which is the best action. Therefore it is a deterministic policy. 
 The critic learns to evaluate the optimal action-value function by using the actors best believed action. 
 To help optimization, noise to the action-vector is used to help exploration. 
 
 ## MADDPG
 
Multi-agent deep deterministic policy gradient builds on ddpg but extends the critic. 
It is build as a deep reinforcement learning method for multi-agent domains.  The difficulty of  traditional algorithms 
in the multi-agent case is: 
 - Q-learning is challenged by an inherent non-stationarity of the environment, 
 - while policy gradient suffers from a variance that increases as the number of agents grows. 
 
The MADDPG actor-critic method is adapted by considering action policies of other agents and therefore becomes able to
successfully learn policies that require complex multi-agent coordination.  Furthermore, the algorithm can leverage a 
training regimen utilizing an ensemble of policies for each agent that leads to more robust multi-agent policies. 
It is successfully shown the strength of our approach compared to existing methods in cooperative as well as competitive
 scenarios, in which the agent populations are able to discover various physical and informational coordination 
 strategies. 

## Approach: 

Algorithmic choice: 


Tennis is normally a zero-sum game without collaborative elements, since the person provoking an error gets a score, 
even if a ball is never returned. 
Such a zero-sum game could be solved by alphazero. 
But, this tennis game mostly rewards getting the ball over the net, which is collaborative. Only a (comparably small) 
score is given to the agent that loses the ball. There is no reward for the opponents failure. 

Reinforcement Learning in adversarial games means that the opponent not only tackles a difficult (but static) environment, 
but the agent has to compete against an improving (non-static) agent. Therefore, a reward is always relative to the 
opponent it is achieved against. 
Using self-play, the opponent is of the same strength. This relatively steady agent provides a smooth learning. But, 
it is difficult to avoid learning a specific policy which is only valid in this self-play but can be easily exploited by 
other agents or humans. 
This can be mitigated by playing against different past policies, or policies with different reward structure. 
The balance between a smooth learning and a diverse and generally good strategy is fundamental in this setting. 

A value-based method would suffer from discretization problems in the continuous action space (right/left and jump). 
A policy-gradient method is already quite unstable in a static environment. Since the self-play still adds instability,
the progress would be difficult, yet probably achievable. OpenAI showed how much a multi-agent PPO agent can learn by
playing DOTA against professional players. 
Actor-critic methods stabilize policy-gradient methods by approximating the value of the states, with a value-based critic. 
This critic stabilizes the advantage (difference of state values plus received reward) by providing an unbiased, but 
stable estimate instead of true, noisy value derived from the individual trajectory or td-step. 

Though MA-PPO has shown very promising results especially at playing DOTA, the MADDPG algorithm is a very promising 
extension to the DDPG actor-critic method. 
The actor still receives the state from the agent's perspective, while the critic evaluates the entire state and the 
output of all the other agents. 
Therefore, the dynamic environment is incorporated into the value estimation of the critic. Because each agent/critic
pair is still trained individually, individual objectives can be learned. 
In this tennis match, the actions are mirrored. Therefore, the agent can in principle play against itself. 


For this environment, two agents were trained using ddpg as a baseline and maddpg as an supposedly improved approach. 
The ddpg reference implementation uses a shared equiprobable experience replay. Each agent is trained with the same
respective perspective of a shared experience. 
The noise is modeled with Ornstein-Uhlenbeck noise, which is reset after each episode. Thus, each episode is augmented with 
biased noise. It is questionable, whether this directional noise is optimal for this environment though, because it is 
very difficult to get a positive reward in the beginning, when the ball is dropped above the agent, but the agent is moving
into a random direction guided by the noise. The training is highly sensitive to this noise. 
This noise is decayed by multiplying it with a coefficient epsilon. 

The hyperparameters are based on the papers, but modified experimentally. Especially the OU-noise is explored through a 
long set of trial and error. 
The network is made as small as possible, to pick up quickly on a thin reward signal. This seems to limit the maximum
possible score. 

The MADDPG agent does not yet converge. The model is also small, though the critic has to incorporate twice the input,
therefore both agent and critic are bigger networks. The agent picks up signal, but never learns enough to truly 
learn to play tennis. 
The MADDPG agent can optionally use a prioritized experience replay, but also relies on Ornstein-Uhlenbeck noise. 

The future ideas section lists the concepts for future improvements. 

## Results: 

With two DDPG agents consisting of seperately trained actor-critic agents. They learn of the same experience replay 
The currently used parameter set can be found in parameters.py of the ddpg directory. 
These parameters are as follows: 

- max_score: 0.8442000127956271
- at_iteration: 1299
- buffer_size: 1000000
- batch_size: 256
- gamma: 0.9
- tau: 0.001
- lr_actor: 0.001
- lr_critic: 0.001
- weight_decay: 0
- ou_mu: 0.0
- ou_theta: 0.15
- ou_sigma: 0.01
- actor_fc1_units: 32
- actor_fc2_units: 16
- critic_fcs1_units: 32
- critic_fc2_units: 16
- random_seed: 15
- update_every: 1
- epsilon: 1.0
- epsilon_decay: 1
- num_episodes: 5000

The Actor looks as follows:

        input = 24 states
        32 Fully Connected Neurons with Batch Normalization and Relu Activation
        16 Fully Connected Neurons with Batch Normalization and Relu Activation
        2 Fully Connected Neurons with Batch Normalization and tanh Activation
      
The Critic looks as follows: 

        input = 24 states
        32 Fully Connected Neurons with Batch Normalization and Relu Activation
        16 Fully Connected Neurons with Batch Normalization and Relu Activation + 2 Action Values of the Actor 
        1 Fully Connected Neuron with Batch Normalization and no Activation
      
These parameters are empirically optimized. It is noteworthy, that such a small neural network can learn this complex 
tennis gameplay. The resulting network weights are derived from episode 1300. 

![Alt text](results/ddpg/1589925396.1851904.png?raw=true "Title")

## Future Ideas: 

As this environment is a mix of competition and collaboration, yet not a zero-sum game, it is ideal for MADDPG. 
But this algorithm does not yield the desired results due to the complex parameter optimization. 
It is implemented with prioritized experience replay, but so far no suitable parameters are found. 
A bug may also be the cause of the poor results depicted below. 

While the network architecture is the same as the above described, the parameters are as follows: 

        # Learning Parameters
        self.buffer_size = int(1e6)  # replay buffer size
        self.batch_size = 64  # minibatch size
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters
        self.lr_actor = 1e-4  # learning rate of the actor
        self.lr_critic = 1e-3  # learning rate of the critic
        self.weight_decay = 1e-2  # L2 weight decay

        # ou noise
        self.ou_mu = 0.
        self.ou_theta = 0.5
        self.ou_sigma = 0.2
        
        # network architecture for actor and critic
        self.actor_fc1_units = 96
        self.actor_fc2_units = 64
        self.critic_fcs1_units = 96
        self.critic_fc2_units = 64

        # Further parameter not found in paper
        self.random_seed = 15  # random seed
        self.update_every = 16  # time steps between updates
        self.num_updates = 1  # num of update passes when updating
        self.epsilon = 1.0  # epsilon for the noise process added to the actions
        self.epsilon_decay = 0.9999  # 1e-6  # decay for epsilon above
        self.num_episodes = 20000  # number of episodes
        self.file_name = 'Tennis_Linux/Tennis.x86_64'
        self.file_name_watch = self.file_name
        self.train = True
        # tuned parameter to "reach" the goal
        # Learning
        self.batch_size = 512  # mini batch size
        self.lr_actor = 1e-4  # learning rate of the actor
        self.lr_critic = 1e-4  # learning rate of the critic
        self.tau = 1e-2

        # Prioritized Experience Replay
        self.use_prioritized_experience_replay = True
        self.per_max_priority = 1.0
        self.per_alpha = 0.4
        self.per_alpha_end = 0.4
        self.per_beta = 0.4
        self.per_beta_end = 1.0
        self.per_annihilation = 8000
        self.per_eps = 1e-3


The results are not solving the environment and need further tuning. 

![Alt text](results/maddpg/1589672949.192302.png?raw=true "Title")

Ideas for both ddpg and maddpg: 

    To tune the parameters automatically, the policy-based algorithm family of hillclimbing methods can be used. 
    Parameter Noise has been proven to perform better than action noise. https://vimeo.com/252185862https://github.com/jvmancuso/ParamNoise
    The memory can be structured along priority: https://github.com/Damcy/prioritized-experience-replay
    Different replay buffer for actor/critic 
    Adding dropouts in critic network to avoid overfitting with a slightly too big network architecture
    Turn off OU noise and use random noise. OU noise does not seem to work all too well in this environment. 


