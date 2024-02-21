import diffuser.sampling as sampling
import diffuser.utils as utils
import numpy as np
import diffuser
import datetime 
from diffuser.models.temporal import LongDynamics

import torch
def cycle(dl):
    while True:
        for data in dl:
            yield data


def eval_diffusion(diffusion, dataset, args):
    policy_config = utils.Config(
    'sampling.policies.NonGuidedPolicy',
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    sample_fn=diffuser.models.diffusion.default_sample_fn,
    verbose=False,
    )

    policy = policy_config()

    #-----------------------------------------------------------------------------#
    #-------------------------------- test dataset -------------------------------#
    #-----------------------------------------------------------------------------#
    dataset_name =args.dataset + "_test"
    horizon = args.horizon
    dataset_config = utils.Config(
       'datasets.SequenceDataset',
        savepath=(args.savepath, 'dataset_config.pkl'),
        env=dataset_name,
        horizon=horizon,
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
    total_delta_actions= 0
    frames = []
    eval_episodes = 30
    total_l2 = 0
    max_l2 = -np.inf

    if diffusion.__class__.__name__ == 'GaussianDiffusion':
        for t in range(eval_episodes):


            ## save state for rendering only
            batch = next(dataloader)
            

            ## format current observation for conditioning
            obs = batch[1][0].squeeze(0)
            conditions = {0: obs}
            action, samples = policy(conditions, batch_size=64, verbose=True)
            pred_observation = samples.observations[0]
            target_observation =  batch[0][0][:,1:].numpy()
            pred_actions = samples.actions[0]
            target_actions = batch[0][0][:,0].numpy()
            #l2 actions
            delta_actions = np.mean(np.abs(pred_actions - target_actions))
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
            mean_l2 = total_l2 / (t + 1)  
            total_delta_actions += delta_actions  
            
            print(
                f'batch: {t} | current l2: {l2:.2f} |  delta_s: {s_delta:.2f} | mean l2: {mean_l2:.2f} | max l2: {current_max_l2:.2f} |',
                flush=True,
            )
            import pdb; pdb.set_trace()
        print("pred_observation s: ",pred_observation[:,0])
        print("target_observation s: ",target_observation[:,0])

        info = {
            "mean_l2": mean_l2,
            "max_l2": max_l2,
            "s_delta": total_delta_s / eval_episodes,
            "v_delta": total_delta_v / eval_episodes,
            "a_delta": total_delta_a / eval_episodes,
            "j_delta": total_delta_j / eval_episodes,
            "action_delta": total_delta_actions / eval_episodes,
        }
        import pdb; pdb.set_trace()
        return info
    
    elif diffusion.__class__.__name__ == 'ActionGaussianDiffusion':
        dynamics = LongDynamics()
        eval_episodes = 50
        total_a0_delta = 0
        for t in range(eval_episodes):
            print(f"episode: {t}/{eval_episodes}", flush=True)
            ## save state for rendering only
            batch = next(dataloader)
            
            pred_action_list = []
            pred_state_list = []
            ## format current observation for conditioning
            obs = batch[1][0].squeeze(0)
            extra_obs = obs.clone()
            conditions = {0: obs}
            #obs = obs.numpy()
            pred_action = policy(conditions, batch_size=1, verbose=True)
            target_action = batch[0][0][:,0].numpy()[0]
            delta_actions= np.mean(np.abs(pred_action - target_action))
            total_a0_delta += delta_actions
            for i in range(0, horizon):
                pred_action = policy(conditions, batch_size=256, verbose=True)
                pred_action_list.append(pred_action[0])
                obs = torch.tensor(obs, dtype=torch.float32)
                pred_action = torch.tensor(pred_action, dtype=torch.float32)
                obs = dynamics(obs[0:4].unsqueeze(0), pred_action[0][0])
                obs = np.concatenate([obs[0], extra_obs[4:]], axis=0)
                pred_state_list.append(obs)
                conditions = {0: obs}
            pred_action_list = np.array(pred_action_list).reshape(horizon, 1)
            pred_state_list = np.array(pred_state_list)
            delta_actions = np.mean(np.abs(pred_action_list - batch[0][0][:,0].numpy()))
            a0_delta = np.mean(np.abs(pred_action_list[0] - batch[0][0][:,0].numpy()[0]))
            total_a0_delta += a0_delta
            diff = np.abs(pred_state_list - batch[0][0][:,1:].numpy())
            s_delta = np.mean(diff[:,0])
            v_delta = np.mean(diff[:,1])
            a_delta = np.mean(diff[:,2])
            j_delta = np.mean(diff[:,3])
            l2 = np.mean(diff)
            total_delta_actions += delta_actions
            total_l2 += l2
            total_delta_a += a_delta
            total_delta_v += v_delta
            total_delta_s += s_delta
            total_delta_j += j_delta

        info = {
            "a0_delta": total_a0_delta / eval_episodes,
            "action_delta": total_delta_actions / eval_episodes,
            "state_l2": total_l2 / eval_episodes,
            "s_delta": total_delta_s / eval_episodes,
            "v_delta": total_delta_v / eval_episodes,
            "a_delta": total_delta_a / eval_episodes,
            "j_delta": total_delta_j / eval_episodes,
        }
        # info = {
        #     "action_delta": total_a0_delta/eval_episodes,
        # }
        print(info)
        return info