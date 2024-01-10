from diffuser.environments.safe_grid import Safe_Grid_v1
import numpy as np
env = Safe_Grid_v1()
state = env.reset()
done = False
dif_x = np.linalg.norm(state[2]-state[0])
dif_y = np.linalg.norm(state[3]-state[1])
ite = 0
while not done:
    if dif_x > 3:
        action_x = 1.5
    elif dif_x > 2.9:
        action_x = state[2]-state[0]
    else:
        action_x = 0
    if dif_y > 3:
        action_y = 1.5
    elif dif_y > 2.9:
        action_y = state[3]-state[1]
    else:
        action_y = 0
    # state[0] = state[0] + action_x
    # state[1] = state[1] + action_y

    dif_x = state[2]-state[0]
    dif_y = state[3]-state[1]
    ite += 1
    action = np.array([action_x,action_y])
    print(f" action: {action}")
    state, reward, cost, done, info = env.step(action)

    print(f"{ite}) state: {state}, reward: {reward}, done: {done}, info: {info}")