import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import numpy as np
import diffuser
import datetime
from diffuser.environments.safe_grid import Safe_Grid, Safe_Grid_v1, Safe_Grid_Simple, Safe_Grid_v2
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')


#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer




## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    'sampling.policies.NonGuidedPolicy',
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=diffuser.models.diffusion.default_sample_fn,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

policy = policy_config()


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
num_episodes = 5   
episode_rewards = [0 for _ in range(num_episodes)]
episode_costs = [0 for _ in range(num_episodes)]

envs = [Safe_Grid_v2(max_number_steps=50) for _ in range(num_episodes)]
obs = [envs[i].reset() for i in range(num_episodes)]
obs = np.array(obs)
dones = [0 for _ in range(num_episodes)]
total_violations = 0
success_rate = 0

while sum(dones) <  num_episodes:

    conditions = {0: obs}
    action, samples = policy(conditions, batch_size=num_episodes, verbose=args.verbose)

    obs_list = []
    t =0
    for i in range(num_episodes):
        obs,reward,cost,done,info=envs[i].step(action[i])
        obs_list.append(obs[None])
        if done:
            if dones[i] == 1:
                pass
            else:
                dones[i] = 1
                episode_rewards[i] += reward
                episode_costs[i] += cost
                total_violations += info['violations']
                now = datetime.datetime.now()
                envs[i].render(path="/home/fernandi/projects/diffuser/imgs", name=f"plan_not_Guided_{now.strftime('%Y-%m-%d_%H:%M:%S')}")
                print(f"Episode {i} finished with reward {episode_rewards[i]} and cost {episode_costs[i]} at timestep {t}")

                if info['goal_reached']:
                    success_rate += 1
        else:
            if dones[i] == 1:
                pass
            else:
                episode_rewards[i] += reward
                episode_costs[i] += cost
                total_violations += info['violations']
        t += 1

    obs = np.concatenate(obs_list, axis=0)



avg_reward = np.mean(episode_rewards)
avg_cost = np.mean(episode_costs)
success_rate = success_rate/num_episodes
print(f"Average reward: {avg_reward}")
print(f"Average cost: {avg_cost}")
print(f"Success rate: {success_rate}")
print(f"Total violations: {total_violations}")



