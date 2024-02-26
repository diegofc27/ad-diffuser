import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.models.temporal import ValueFunctionL2, LongDynamics
import numpy as np
import datetime
from diffuser.environments.safe_grid import  Safe_Grid_v2
import json
import torch
from PIL import Image
import os
from pathlib import Path
#-----------------------
# ------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#
def make_gif(img_folder: Path, path_gif, num):
    frames = []
    for i in range(0, num):
        filename = img_folder / f"planning_{i}.png"
        frames.append(Image.open(filename))
        # save image
        # im = Image.open(filename)
        # im.save(f"/home/fernandi/projects/bmw23/data/imitation/imgs/paper/{filename.stem}.png", dpi=(900,900))
        # #save img with higher dpi
        # im.save(f"/home/fernandi/projects/bmw23/data/imitation/imgs/paper/{filename.stem}_high_dpi.png", subsampling=0, quality=100)
        # #save as pdf
        # im.save(f"/home/fernandi/projects/bmw23/data/imitation/imgs/paper/{filename.stem}.pdf", dpi=(900,900))

        # plt.savefig(filename, dpi=300)

    frame_one = frames[0]
    frame_one.save(
        path_gif,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=120,
        loop=0,
    )
    
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
    
batch = next(dataloader)
obs = batch[1][0].squeeze(0)
conditions = {0: obs}
# sample_kwargs = {"return_chain":True}

action, samples, chains = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
pred_observation = samples.observations[0]
pred_actions = samples.actions[0]
target_actions = batch[0][0][:,0].numpy()
target_observation =  batch[0][0][:,1:].numpy()
delta_actions = np.mean(np.abs(pred_actions - target_actions))
total_delta_actions += delta_actions
total_diff = 0

diff = np.abs(pred_observation - target_observation)
s_delta = np.mean(diff[:,0])
v_delta = np.mean(diff[:,1])
a_delta = np.mean(diff[:,2])
j_delta = np.mean(diff[:,3])
l2 = np.mean(diff)

import matplotlib.pyplot as plt
# for i in range(args.n_diffusion_steps +1):
#     pred_observation = chains[i,:,0]
#     plt.title(f"Position Plan. Denousing step: {i} - {args.n_diffusion_steps}")
#     plt.xlabel("Time Step")
#     plt.ylabel("Position")
#     plt.xlim(0, 32)
#     plt.ylim(-60, 250)
#     plt.plot(pred_observation, label="Diffusion Plan")
#     plt.plot(target_observation[:,0], label="MPC Target")
#     plt.legend(loc='upper left')
#     #keep legend in upper left corner
#     plt.tight_layout()
#     plt.grid(True)
#     plt.show()

#     plt.savefig(f"/home/fernandi/projects/diffuser/scripts/plots/planning_{i}.png")
#     #clear plot
#     plt.clf()
# for i in range(20):
#     pred_observation = chains[20,:,0]
#     plt.title(f"Position Plan. Denousing step: {20} - {args.n_diffusion_steps}")
#     plt.xlabel("Time Step")
#     plt.ylabel("Position")
#     plt.xlim(0, 32)
#     plt.ylim(-60, 250)
#     plt.plot(pred_observation, label="Diffusion Plan")
#     plt.plot(target_observation[:,0],label="MPC Target")
#     plt.legend(loc='upper left')
#     #keep legend in upper left corner
#     plt.tight_layout()
#     plt.grid(True)
#     plt.show()

#     plt.savefig(f"/home/fernandi/projects/diffuser/scripts/plots/planning_{args.n_diffusion_steps+i}.png")
#     #clear plot
#     plt.clf()
# make_gif(Path("/home/fernandi/projects/diffuser/scripts/plots/"), "/home/fernandi/projects/diffuser/scripts/plots/planning.gif", 39 +1)

#plot 4 plots for position, velocity, acceleration and jerk next to each other

min_s = np.min(target_observation[:,0])
max_s = np.max(target_observation[:,0])
min_v = np.min(target_observation[:,1])
max_v = np.max(target_observation[:,1])
min_a = np.min(target_observation[:,2])
max_a = np.max(target_observation[:,2])
min_j = np.min(target_observation[:,3])
max_j = np.max(target_observation[:,3])
offset = [50,20,5, 5]
for i in range(args.n_diffusion_steps +1):
    plt.title(f"Diffusion Plan. Denousing step: {i} - {args.n_diffusion_steps}")
    fig, axs = plt.subplots(1, 4, sharex=True)
    fig.set_size_inches(18, 5)
    #add title
    fig.suptitle(f"Diffusion Plan. Denousing step: {i} - {args.n_diffusion_steps}")
    axs[0].set_ylabel(r"Position [m]")
    axs[1].set_ylabel(r"Velocity [m/s]")
    axs[2].set_ylabel(r"Acceleration [$m/s^2$]")
    axs[3].set_ylabel(r"Control [$m/s^4$]")

    for j in range(4):
        axs[j].set_xlabel(r"Time Step [s]")
        axs[j].grid(True)
    pred_observation = chains[i,:,0]
    pred_velocity = chains[i,:,1]
    pred_acceleration = chains[i,:,2]
    pred_jerk = chains[i,:,3]
    axs[0].set_ylim(min_s - offset[0], max_s + offset[0])
    x = np.arange(0, 32, 1) * 0.2
    axs[0].plot(x,pred_observation, label=f"Diffusion Prediction")
    axs[0].plot(x,target_observation[:,0], label="MPC Target")
    axs[0].plot(x, target_observation[:,4], label="Lead Vehicle", linestyle="--")
    axs[1].set_ylim(min_v - offset[1], max_v + offset[1])
    axs[1].plot(x,pred_velocity, label=f"Diffusion Prediction")
    axs[1].plot(x,target_observation[:,1], label="MPC Target")
    axs[2].set_ylim(min_a - offset[2], max_a + offset[2])
    axs[2].plot(x,pred_acceleration, label=f"Diffusion Prediction")
    axs[2].plot(x,target_observation[:,2], label="MPC Target")
    axs[3].set_ylim(min_j - offset[3], max_j + offset[3])
    axs[3].plot(x,pred_jerk, label=f"Diffusion Prediction")
    axs[3].plot(x,target_observation[:,3], label="MPC Target")
    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')
    axs[2].legend(loc='upper left')
    axs[3].legend(loc='upper left')
    plt.show()
    
    plt.tight_layout()
    plt.savefig(f"/home/fernandi/projects/diffuser/scripts/plots/planning_{i}.png")

for i in range(20):
    plt.title(f"Diffusion Plan. Denousing step: {20} - {args.n_diffusion_steps}")
    fig, axs = plt.subplots(1, 4, sharex=True)
    fig.suptitle(f"Diffusion Plan. Denousing step: {20} - {args.n_diffusion_steps}")
    fig.set_size_inches(18, 5)
    axs[0].set_ylabel(r"Position [m]")
    axs[1].set_ylabel(r"Velocity [m/s]")
    axs[2].set_ylabel(r"Acceleration [$m/s^2$]")
    axs[3].set_ylabel(r"Control [$m/s^4$]")

    for j in range(4):
        axs[j].set_xlabel(r"Time Step [s]")
        axs[j].grid(True)
    pred_observation = chains[20,:,0]
    pred_velocity = chains[20,:,1]
    pred_acceleration = chains[20,:,2]
    pred_jerk = chains[20,:,3]
    x = np.arange(0, 32, 1) * 0.2
    axs[0].set_ylim(min_s - offset[0], max_s + offset[0])
    axs[0].plot(x, pred_observation, label=f"Diffusion Prediction")
    axs[0].plot(x, target_observation[:,0], label="MPC Target")
    axs[0].plot(x, target_observation[:,4], label="Lead Vehicle", linestyle="--")
    axs[1].set_ylim(min_v - offset[1], max_v + offset[1])
    axs[1].plot(x, pred_velocity, label=f"Diffusion Prediction")
    axs[1].plot(x, target_observation[:,1], label="MPC Target")
    axs[2].set_ylim(min_a - offset[2], max_a + offset[2])
    axs[2].plot(x, pred_acceleration, label=f"Diffusion Prediction")
    axs[2].plot(x, target_observation[:,2], label="MPC Target")
    axs[3].set_ylim(min_j - offset[3], max_j + offset[3])
    axs[3].plot(x, pred_jerk, label=f"Diffusion Prediction")
    axs[3].plot(x, target_observation[:,3], label="MPC Target")
    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')
    axs[2].legend(loc='upper left')
    axs[3].legend(loc='upper left')
    plt.show()
    
    plt.tight_layout()
    plt.savefig(f"/home/fernandi/projects/diffuser/scripts/plots/planning_{20+i}.png")
    #clear plot
make_gif(Path("/home/fernandi/projects/diffuser/scripts/plots/"), "/home/fernandi/projects/diffuser/scripts/plots/planning.gif", 39 +1)


