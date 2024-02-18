import os
import collections
import numpy as np
import gym
import pdb
import pickle
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

from diffuser.environments.safe_grid import Safe_Grid, Safe_Grid_v1, Safe_Grid_Simple, Safe_Grid_v2
@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# with suppress_output():
#     # d4rl prints out a variety of warnings
#     import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if name == 'Safe_Grid_Simple-v0':
        env = Safe_Grid_Simple()
        return env
    elif name == 'Safe_Grid-v1':
        env = Safe_Grid_v1()
        return env
    elif name == 'SafeGrid-v2':
        env = Safe_Grid_v2()
        return env
    elif name == 'Safe_Grid-v0':
        env = Safe_Grid()
        return env
    elif name == 'BMW-v0' or name == 'BMW-v0_test':
        class BMWEnv(gym.Env):
            def __init__(self):
                self.action_space = gym.spaces.Box(-1, 1, (2,))
                self.observation_space = gym.spaces.Box(-1, 1, (4,))
                self.max_episode_steps = 32
                self._max_episode_steps = 32
                self.name = name
            def reset(self):
                return np.zeros(4)
            def step(self, action):
                return np.zeros(4), 0, True, {}
        return BMWEnv()
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    if env.name == 'SafeGrid-v2':
         #iold = '/home/fernandi/projects/diffuser/trajectories/safe_grid_v2_10000__rate_0.92.pickle'
         with open('/home/fernandi/projects/diffuser/trajectories/safe_grid_v2_10000__rate_0.92.pickle', 'rb') as handle:
            dataset = pickle.load(handle)
            print("loaded pickle")
    elif env.name == 'BMW-v0':
        with open('/home/fernandi/projects/diffuser/trajectories/bmw_norm_train.pkl', 'rb') as handle:
            dataset = pickle.load(handle)
    elif env.name == 'BMW-v0_test':
        with open('/home/fernandi/projects/diffuser/trajectories/bmw_norm_test.pkl', 'rb') as handle:
            dataset = pickle.load(handle)
    
    elif env.name == 'SafeGrid-v1' or env.name == 'Safe_Grid_Simple-v0':
         with open('/home/fernandi/projects/decision-diffuser/code/trajectories/safe_grid_v1_5000__rate_0.93.pickle', 'rb') as handle:
            dataset = pickle.load(handle)
            print("loaded pickle")

    elif(env.unwrapped.spec.id=='CartPole-v1'):
        with open('/home/fernandi/projects/diffuser/history/cartpole/history.pickle', 'rb') as handle:
            dataset = pickle.load(handle)
            print("loaded pickle")
    else:
        dataset = env.get_dataset()
    print("episodes")
    print((dataset['terminals']==True).sum())
    if 'antmaze' in str(env).lower():
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

    return dataset

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset
    c = 0
    episode_step = 0
    idx = 0
    idx2= 0
    for i in range(N):
        # print("i",i)
        # print("c ",c)
        # c+=1
        
        done_bool = bool(dataset['terminals'][i])
        

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
            #print("final",final_timestep )
        else:
            final_timestep = False
            #final_timestep = (episode_step == env._max_episode_steps - 1)
            # final_timestep = (episode_step == env._max_episode_steps)
            # print("final",final_timestep, "episode step", i) 
            
        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])
        #if done_bool:
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
