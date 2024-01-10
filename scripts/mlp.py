
import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.utils.data import Dataset
import wandb
from diffuser.environments.safe_grid import Safe_Grid, Safe_Grid_v1

class GridDataset(Dataset):
    def __init__(self, path):
        self.data =   pickle.load(open(path, "rb"))

    def __len__(self):
        return self.data["observations"].shape[0]

    def __getitem__(self, idx):    
        X = self.data["observations"][idx]
        X = torch.tensor(X, dtype=torch.float32)
        y = self.data["actions"][idx]
        y = torch.tensor(y, dtype=torch.float32)
        return X,y
    

def wandb_init() -> None:
    wandb.init(
        config=[],
        project="thesis",
        group="testMLP",
        name="testMLP",
        entity="diegofc77",

    )
    wandb.run.save()

def main():
    path ="/home/fernandi/projects/diffuser/trajectories/safe_grid_v1_5000__rate_0.93.pickle"
    
    #create dataloader
    device="cuda:4"
    dataset = GridDataset(path)
    # wandb_init()
    #sample 
    loss_fn = nn.L1Loss()

    model = nn.Sequential(
        nn.Linear(14, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 2),
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(10000000, model, dataset, loss_fn, optimizer,32,device)

@torch.no_grad()
def eval(
    env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env = Safe_Grid_v1()
    episode_rewards = []
    episode_costs = []
    success_rate = 0.0
    for e in range(n_episodes):

        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_cost = 0.0
        while not done:
            action = actor(torch.tensor(state,device=device, dtype=torch.float32)).cpu().numpy()
            #state, reward, done, _ = env.step(torch.tensor(action))
            state, reward, cost, done, info = env.step(action)
            episode_reward += reward
            episode_cost += cost
            if info["goal_reached"]: success_rate+=1 
        env.render(path="/home/fernandi/projects/diffuser/imgs/mlp", name=f"safe_grid_data{e}")

        env.close()
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        # print(f"Episode reward: {episode_reward:.3f}, Episode cost: {episode_cost:.3f}")

    return np.asarray(episode_rewards), np.asarray(episode_costs), np.asarray(success_rate/n_episodes)
def train(num_timesteps, model, dataset, loss_fn, optimizer, batch_size=32, device="cuda:1"):
    loss_history = []
    for n in range(num_timesteps):
        indices = np.random.randint(0, len(dataset), size=batch_size)
        X, y = dataset[indices]
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
       
        loss = loss_fn(y_pred, y)
        log = {"loss": loss.item()}
        #wandb.log(log, step=n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if n % 1000 == 0:
            print(f"step: {n} | loss: {loss.item()}")
        if n % 10000 == 0:
            rewards, costs, success_rate = eval(None, model, device, 50, 0)
            log = {"eval_reward": np.mean(rewards), "eval_cost": np.mean(costs), "success_rate": success_rate}
            print(log)
            torch.save(model.state_dict(), "/home/fernandi/projects/diffuser/runs/mlp/mlp.pt")
            #

    #plot loss
    


if __name__ == "__main__":
    main()