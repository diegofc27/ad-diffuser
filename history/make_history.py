import numpy as np

#load files
observations = np.load('observations.npy')
actions= np.load('actions.npy')
rewards = np.load('rewards.npy')
dones = np.load('dones.npy')
#check if dim are correct
observations = observations.reshape(-1,4)
#create and save dict pickle
assert observations.shape[0] == actions.shape[0], "dim are not correct"

history = {"observations":observations, "actions":actions,"rewards":rewards,"terminals":dones}

import pickle

with open('history.pickle', 'wb') as handle:
    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

#2851