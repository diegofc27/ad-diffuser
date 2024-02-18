import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.models.temporal import ValueFunctionL2, LongDynamics
import numpy as np
import datetime
from diffuser.environments.safe_grid import  Safe_Grid_v2
import json
import torch
import os
#-----------------------
# ------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan_l2_bmw')

l2_guide = False

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#
save_path = f"/home/fernandi/projects/diffuser/{args.loadbase}/{args.dataset}/{args.value_loadpath}"
## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
# if not l2_guide:
#     value_experiment = utils.load_diffusion(
#         args.loadbase, args.dataset, args.value_loadpath,
#         epoch=args.value_epoch, seed=args.seed,
#     )

#     ## ensure that the diffusion model and value function are compatible with each other
#     utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guide
if l2_guide:
    value_function = ValueFunctionL2()
    guide_config = utils.Config(args.guide, model=value_function, verbose=False)
    guide = guide_config()
    name ="results_guided"
else:
    # value_function = value_experiment.ema
    name = "results_unguided"


logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## policies are wrappers around an unconditional diffusion model and a value guide

if l2_guide:
    policy_config = utils.Config(
        args.policy,
        guide=guide,
        scale=args.scale,
        diffusion_model=diffusion,
        normalizer=dataset.normalizer,
        preprocess_fns=args.preprocess_fns,
        ## sampling kwargs
        sample_fn=sampling.n_step_guided_p_sample,
        n_guide_steps=args.n_guide_steps,
        t_stopgrad=args.t_stopgrad,
        scale_grad_by_std=args.scale_grad_by_std,
        verbose=False,
    )
else:
    policy_config = utils.Config(
    'sampling.policies.NonGuidedPolicy',
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    sample_fn=sampling.n_step_guided_p_sample,
    verbose=False,
    )
logger = logger_config()
policy = policy_config()

dataset_config = utils.Config(
        'datasets.SequenceDataset',
        savepath=(args.savepath, 'dataset_config.pkl'),
        env=args.dataset + "_test",
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        use_normalizer=False
    )

test_dataset = dataset_config()
dataloader = cycle(torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=4, shuffle=True, pin_memory=True
    ))


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
total_reward = 0
total_cost = 0
total_delta_s = 0
total_delta_v = 0
total_delta_a = 0
total_delta_j = 0
frames = []
eval_episodes = 200
total_l2 = 0
max_l2 = -np.inf
total_diff_dyn_s = 0
total_diff_dyn_v = 0
total_diff_dyn_a = 0
total_diff_dyn_j = 0
total_delta_actions= 0
dynamics = LongDynamics()
for i in range(eval_episodes):
    
    batch = next(dataloader)
    obs = batch[1][0].squeeze(0)
    conditions = {0: obs}
 
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
    pred_observation = samples.observations[0]
    pred_actions = samples.actions[0]
    target_actions = batch[0][0][:,0].numpy()
    target_observation =  batch[0][0][:,1:].numpy()

    delta_actions = np.mean(np.abs(pred_actions - target_actions))
    total_delta_actions += delta_actions
    total_diff = 0
    for j in range(31):
        state = torch.tensor(pred_observation[j,:4], device="cuda:0").unsqueeze(0)
        action = torch.tensor(samples.actions[0][j], device="cuda:0")
        new_state = dynamics.forward(state,action)
        real_state = target_observation[j+1,:4].copy()
        diff = torch.abs(new_state.cpu() - real_state)
        total_diff += diff
    avg_diff = total_diff / 31
    ## l2 loss
    diff = np.abs(pred_observation - target_observation)
    s_delta = np.mean(diff[:,0])
    v_delta = np.mean(diff[:,1])
    a_delta = np.mean(diff[:,2])
    j_delta = np.mean(diff[:,3])
    l2 = np.mean(diff)
    current_max_l2 = np.max(diff)

    if current_max_l2 > max_l2:
        max_l2 = current_max_l2
    total_l2 += l2
    total_delta_a += a_delta
    total_delta_v += v_delta
    total_delta_s += s_delta
    total_delta_j += j_delta
    mean_l2 = total_l2 / (i + 1)    
    diff_dyn_s = avg_diff[0][0].item()
    diff_dyn_v = avg_diff[0][1].item()
    diff_dyn_a = avg_diff[0][2].item()
    diff_dyn_j = avg_diff[0][3].item()
    total_diff_dyn_s += diff_dyn_s
    total_diff_dyn_v += diff_dyn_v
    total_diff_dyn_a += diff_dyn_a
    total_diff_dyn_j += diff_dyn_j


    print(
        f'batch: {i} | current l2: {l2:.2f} | avg_delta_action {total_delta_actions / (i + 1):.2f} avg_delta_s: {total_delta_s/(i + 1) :.2f} | dynamics_diff_s: {total_diff_dyn_s/(i + 1):.2f} | mean l2: {mean_l2:.2f} | max l2: {current_max_l2:.2f} |',
        flush=True,
    )


info = {
    "mean_l2": mean_l2,
    "max_l2": max_l2,
    "action_delta": total_delta_actions / eval_episodes,
    "s_delta": total_delta_s / eval_episodes,
    "v_delta": total_delta_v / eval_episodes,
    "a_delta": total_delta_a / eval_episodes,
    "j_delta": total_delta_j / eval_episodes,
    "dynamics_diff_s": total_diff_dyn_s / eval_episodes,
    "dynamics_diff_v": total_diff_dyn_v / eval_episodes,
    "dynamics_diff_a": total_diff_dyn_a / eval_episodes,
    "dynamics_diff_j": total_diff_dyn_j / eval_episodes,
}
#convert info to str
for k,v in info.items():
    info[k] = str(v)
print("dynamics error",avg_diff / eval_episodes)
print("metrics: ",info)
#save json with values in args.loadbase
if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(save_path + f"/{name}.json", "w") as f:
    json.dump(info,f)

## write results to json file at `args.savepath`
#logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)