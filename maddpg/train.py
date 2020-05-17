import time
from collections import deque
from itertools import count

import numpy as np
from unityagents import UnityEnvironment

from maddpg.maddpg_agent import MADDPG
from maddpg.parameters import Par
from maddpg.plotting import Plotting


def ddpg_unity(env, agent, brain_name, num_agents, plotting, par):
    """Train DDPG Agent
    Params
    ======
        env (object): Unity environment instance
        agent (DDPGMultiAgent): agent instance
        brain_name (string): name of brain
        num_agents (int): number of agents
        plotting (Plotting): object to plot
        n_episodes (int): number of episodes to train the network
    """
    # for naming the weights
    experiment_name = plotting.fname

    not_solved = True
    scores_deque = deque(maxlen=100)
    scores = []
    best_score = -np.Inf
    for i_episode in range(1, par.num_episodes + 1):
        if i_episode % 500 < 5:
            env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        else:
            env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        score = np.zeros(num_agents)  # initialize the score (for each agent)
        t_s = time.time()
        for timestep in count():
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            score += env_info.rewards  # update the score (for each agent)
            agent.step(states, actions, rewards, next_states, dones, timestep)
            states = next_states  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                plotting.add_measurement(score_per_agent=score, eps=agent.epsilon)
                break
        ctime = time.time() - t_s
        scores_deque.append(np.mean(score))
        scores.append(score)
        best_score = max(best_score, np.mean(scores_deque))
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}, Min Score: {:.2f}, Max Score: {:.2f}, '
              'episode_dur {:.2f}'.format(i_episode, np.mean(scores_deque), np.mean(score), np.min(score),
                                          np.max(score), ctime), end="")
        if i_episode % 100 == 0:
            save_network_weights(agent, experiment_name, i_episode)
            plotting.plotting(par)
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}, Min Score: {:.2f}, Max Score: {:.2f}, '
                  'episode_dur {:.2f}'.format(
                i_episode, np.mean(scores_deque), np.mean(score), np.min(score), np.max(score), ctime))
        if np.mean(scores_deque) >= 0.5 and not_solved:
            print("environment solved in {}".format(i_episode))
            not_solved = False

    # save at last
    plotting.plotting(par)
    save_network_weights(agent, experiment_name, par.num_episodes)
    return scores


def save_network_weights(agent, experiment_name, i_episode):
    pass
    # torch.save(agent.actor_local.state_dict(), experiment_name + '_checkpoint_actor_' + str(i_episode) + '.pth')
    # torch.save(agent.critic_local.state_dict(), experiment_name + '_checkpoint_critic_' + str(i_episode) + '.pth')


def train_in_unity_env():
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=par.train)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)

    # size of each action
    action_size = brain.vector_action_space_size
    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {} and has {} actions.'.format(states.shape[0],
                                                                                                  state_size,
                                                                                                  action_size))
    plotting = Plotting()

    # optimize for two agents playing against each other tennis.
    # adaptions needed to distribute among more agents
    # state_size, state_size_full, action_size, par, num_agents, num_instances=1
    agent = MADDPG(state_size, action_size, par, num_agents)

    ddpg_unity(env, agent, brain_name, num_agents, plotting, par)
    env.close()


# main function
if __name__ == "__main__":
    # Get parameters for ddpg from config object
    par = Par()

    env = UnityEnvironment(file_name=par.file_name, no_graphics=False)

    # load and train within the unity environment
    train_in_unity_env()
