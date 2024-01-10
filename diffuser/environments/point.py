#Importing required libraries
import gym
from gym import spaces
import random
import numpy as np
#Creating the custom environment
#Custom environment needs to inherit from the abstract class gym.Env
class Find_Dot(gym.Env):
    #add the metadata attribute to your class
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self,render_mode=None,max_number_steps=20):
        # define the environment's action_space and observation space
        
        '''Box-The argument low specifies the lower bound of each dimension and high specifies the upper bounds
        '''
        self.observation_space= gym.spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float32)
         
        self.action_space= gym.spaces.Box(low=-1,high=1, shape=(2,), dtype=np.float32)
        self.goal_x_rand = 5
        self.goal_y_rand = 20
        self.state= np.array([np.random.uniform(0,5),np.random.uniform(0,5),np.random.uniform(self.goal_x_rand,self.goal_y_rand),np.random.uniform(self.goal_x_rand,self.goal_y_rand)])
        self.goal_distance = .5
        self.current_step =0
        self.max_number_steps =max_number_steps
        self.reward=0
    
    
    def step(self, action):
        '''defines the logic of your environment when the agent takes an actio
        Accepts an action, computes the state of the environment after applying that action
        '''
        done=False
        info={}
      
        #setting the state of the environment based on agent's action
        # rewarding the agent for the action
        action = np.clip(action, -1, 1)
        self.state[0] +=action[0]
        self.state[1] +=action[1]
        distance_to_goal =np.linalg.norm(self.state[0:2]-self.state[2:4])
        self.reward = -np.linalg.norm(self.state[0:2]-self.state[2:4])
        self.current_step +=1
        # define the completion of the episode
 
        if self.current_step>=self.max_number_steps or distance_to_goal<=self.goal_distance:
            done= True
        return self.state, self.reward, done, info
    def render(self,action):
        # Visualize your environment
        print(f"\n Current position:{self.state[0:2]}\n Goal position: {self.state[2:4]} Reward Received:{self.reward} ")
        print(f" Action taken: delta_x: {action[0]}, delta_y: {action[1]}")
        print("==================================================")
    def reset(self):
        #reset your environment
        self.state= np.array([np.random.uniform(0,5),np.random.uniform(0,5),np.random.uniform(self.goal_x_rand,self.goal_y_rand),np.random.uniform(self.goal_x_rand,self.goal_y_rand)])
        self.reward=0
        self.current_step=0
        return self.state
    def close(self):
        # close the nevironment
        self.state=0
        self.reward=0