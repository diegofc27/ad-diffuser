import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
#from diffuser.datasets.d4rl import load_environment
import gymnasium as gym
import numpy as np
import gym
from history.clean_ppo import Agent

agent = Agent(17,6)
agent.load_state_dict(torch.load("/home/fernandi/projects/diffuser/history/HalfCheetah-v2_129000_544.pth"), strict=False)

train_env = gym.make("HalfCheetah-v2")

state = train_env.reset()
done = False
total_reward =0
print(agent.get_action_and_value(torch.rand(1,17)))
while not done:
    state =torch.Tensor(state).unsqueeze(0)
    print(state.shape)
    with torch.no_grad():
        action, logprob, _, value = agent.get_action_and_value(state)
    
    state, reward, done, _  = train_env.step(action.cpu().numpy())
    total_reward+=reward
    print("total_rewards ",total_reward)