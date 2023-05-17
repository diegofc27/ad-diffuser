from pathlib import Path
import pickle
import numpy as np

# #load files
# observations = np.load('observations.npy')
# actions= np.load('actions.npy')
# rewards = np.load('rewards.npy')
# dones = np.load('dones.npy')
# episode_idx = np.load('episode_idx.npy')
# #check if dim are correct
# observations = observations.reshape(-1,4)
# #create and save dict pickle
# assert observations.shape[0] == actions.shape[0], "dim are not correct"

# history = {"observations":observations, "actions":actions,"rewards":rewards,"terminals":dones,"episode_idx":episode_idx}
# import pdb; pdb.set_trace()
# import pickle

# with open('history.pickle', 'wb') as handle:
#     pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

# #2851
path = Path("/home/fernandi/projects/diffuser/history/cartpole/history_raw.pickle")
with open(path, "rb") as fin:
    history = pickle.load(fin)
history['observations'] = np.array(history['observations']).reshape(-1,4)
history['rewards'] = np.array(history['rewards']).reshape(-1,)
history['terminals'] = np.array(history['terminals']).reshape(-1,)
history['actions'] = np.array(history['actions']).reshape(-1,)
history['episode_idx'] = np.array(history['episode_idx']).reshape(-1,)


with open('/home/fernandi/projects/diffuser/history/cartpole/history.pickle', 'wb') as handle:
    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

import pdb; pdb.set_trace()