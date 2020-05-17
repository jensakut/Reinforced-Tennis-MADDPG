#### Udacity Deep Reinforcement Learning Nanodegree 
## Project 2: Competitive Environments
# Reinforced Tennis Playing with MADDPG 
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
    python maddpg/train.py
    python maddpg/watch_reacher.py
    ```



## Results: 

Not Available yet. Currently, something is happening: 

The currently used parameter set can be found in parameters.py 

![Alt text](results/current_results.png?raw=true "Title")

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
 
 
 