# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass

import gym
from gym import wrappers

from diffuser.environments.safe_grid import Safe_Grid_Simple, Safe_Grid, Safe_Grid_v1, Safe_Grid_v2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from typing import Callable
from argparse import ArgumentParser
from tqdm import tqdm
import pickle
from pathlib import Path

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 6854
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "thesis"
    """the wandb's project name"""
    wandb_entity: str = "diegofc77"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Safe_Grid_v2"
    """the id of the environment"""
    total_timesteps: int = 400000000
    """total timesteps of the experiments"""
    log_freq: int = 10
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 300
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
):
    envs = Safe_Grid_v2(max_number_steps=50)
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs = envs.reset()
    episodic_returns = []
    returns =0
    success = 0
    i = 0
    path = f"/home/fernandi/projects/diffuser/imgs/{env_id}/{run_name}"
    #create folder if not exists
    if not os.path.exists(path):
        os.makedirs(path)
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs[None]).to(device))
        next_obs, rewards,cost,terminal, infos = envs.step(actions.cpu().numpy()[0])
        returns += rewards
        i+=1
        if terminal:
            episodic_returns.append(returns)
            if infos["goal_reached"]:
                success += 1
            else:
                print(f"Fail. Number of steps={i}")
            returns=0
            i=0
            envs.render(path=f"/home/fernandi/projects/diffuser/imgs/{env_id}/{run_name}", name=f"ppo_safe_grid_v2{len(episodic_returns)}")
            next_obs = envs.reset()


        obs = next_obs
    print(f"Success rate: {success/eval_episodes}")
    return episodic_returns, success/eval_episodes

def generate_data(
    model_path: str,
    make_env: Callable,
    env_id: str,
    num_trajectories: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
):
    pbar = tqdm(total=num_trajectories)
    envs = Safe_Grid_v2(max_number_steps=50)
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    dataset = defaultdict(list)
    state = envs.reset()
    episodic_returns = []
    returns =0
    success = 0
    i = 0
    path = f"/home/fernandi/projects/diffuser/imgs/{env_id}/{run_name}"
    #create folder if not exists
    if not os.path.exists(path):
        os.makedirs(path)
    while len(episodic_returns) < num_trajectories:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(state[None]).to(device))
        old_state = state.copy()
        next_state, rewards,cost, done, infos = envs.step(actions.cpu().numpy()[0])
        dataset["observations"].append(old_state)
        dataset["actions"].append(actions.cpu().numpy()[0])
        dataset["rewards"].append(rewards)
        dataset["costs"].append(cost)
        dataset["terminals"].append(done)
        returns += rewards
        i+=1
        if done:
            episodic_returns.append(returns)
            if infos["goal_reached"]:
                success += 1
            returns=0
            i=0
            #envs.render(path=f"/home/fernandi/projects/diffuser/imgs/{env_id}/{run_name}", name=f"ppo_safe_grid_v2{len(episodic_returns)}")
            state = envs.reset()
            pbar.update(1)
            # obs_shape = np.array(dataset["observations"]).shape
            # print(f"observations: {obs_shape}")
            # act_shape = np.array(dataset["actions"]).shape
            # print(f"actions: {act_shape}")
            # rew_shape = np.array(dataset["rewards"]).shape
            # print(f"rewards: {rew_shape}")
            # cost_shape = np.array(dataset["costs"]).shape
            # print(f"costs: {cost_shape}")
            # term_shape = np.array(dataset["terminals"]).shape
            # print(f"terminals: {term_shape}")


        state = next_state
    print(f"Success rate: {success/num_trajectories}")


    dataset["observations"] = np.array(dataset["observations"]).reshape(-1,15)
    dataset["actions"] = np.array(dataset["actions"]).reshape(-1,2)
    dataset["rewards"] = np.array(dataset["rewards"]).reshape(-1)
    dataset["costs"] = np.array(dataset["costs"]).reshape(-1)
    dataset["terminals"] = np.array(dataset["terminals"]).reshape(-1)  
    pbar.close()
    #round rate 2 decimals
    rate = round(success/num_trajectories,2)
    path = Path(f"/home/fernandi/projects/diffuser/trajectories/safe_grid_v2_{num_trajectories}__rate_{rate}.pickle")
    with open(path, "wb") as fout:
        pickle.dump(dataset, fout)
    import pdb; pdb.set_trace()
    return episodic_returns, success/num_trajectories

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        env = Safe_Grid_v2(max_number_steps=50)
        # env = wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        # env = wrappers.RecordEpisodeStatistics(env)
        # env = wrappers.ClipAction(env)
        # env = wrappers.NormalizeObservation(env)
        # env = wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = wrappers.NormalizeReward(env, gamma=gamma)
        # env = wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(15).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(15).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            
            layer_init(nn.Linear(512, np.prod(2)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(2)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
   

    args_terminal = parser.parse_args()

    args = Args()

    args.seed = args_terminal.seed
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = "cuda:5"
    run_name = "Safe_Grid_v2_ppo_509_safegymReward_1704739553"
    episodic_returns, sucess_rate = generate_data(
                f"runs/{run_name}/ppo_best.pt",
                make_env,
                args.env_id,
                num_trajectories=100,
                run_name=f"{run_name}-eval",
                Model=Agent,
                device=device,
                gamma=args.gamma,
            )
    import pdb; pdb.set_trace()
    print(f"num_iterations={args.num_iterations}")
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_safegymReward_{int(time.time())}"
    


    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group="with_l2_goal",
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    best_reward = -np.inf

   

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        print(f"-------- iteration {iteration} --------")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, costs, terminations, infos = envs.step(action.cpu().numpy())
            next_done = terminations
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if iteration % args.log_freq == 0:
            model_path = f"runs/{run_name}/{args.exp_name}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
            # evaluate agent
            episodic_returns, sucess_rate = evaluate(
                f"runs/{run_name}/{args.exp_name}.pt",
                make_env,
                args.env_id,
                eval_episodes=50,
                run_name=f"{run_name}-eval",
                Model=Agent,
                device=device,
                gamma=args.gamma,
            )
            if sucess_rate > best_reward:
                best_reward = sucess_rate
                torch.save(agent.state_dict(), f"runs/{run_name}/{args.exp_name}_best.pt")
                print(f"best model saved to runs/{run_name}/{args.exp_name}_best.pt")
            for idx, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, global_step + idx * args.num_envs)
            wandb.log({"eval/episodic_return": np.array(episodic_returns).mean(), "global_step": global_step + idx * args.num_envs})
            wandb.log({"eval/sucess_rate": sucess_rate, "global_step": global_step + idx * args.num_envs})

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     args.env_id,
        #     eval_episodes=10,
        #     run_name=f"{run_name}-eval",
        #     Model=Agent,
        #     device=device,
        #     gamma=args.gamma,
        # )
        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

    
    envs.close()
    writer.close()