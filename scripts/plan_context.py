import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import numpy as np
from diffuser.models.diffusion import default_sample_fn
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan-ad')


#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#
## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
print("DONE LOADING DIFFUSION")
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

if args.guided:
    ## policies are wrappers around an unconditional diffusion model and a value guide
    policy_config = utils.Config(
        args.policy,
        guide=guide,
        scale=args.scale,
        diffusion_model=diffusion,
        normalizer=dataset.normalizer,
        preprocess_fns=args.preprocess_fns,
        ## sampling kwargs
        sample_fn=sampling.functions.n_step_guided_p_sample_all,
        n_guide_steps=args.n_guide_steps,
        t_stopgrad=args.t_stopgrad,
        scale_grad_by_std=args.scale_grad_by_std,
        verbose=False,
    )
else:
    policy_config = utils.Config(
        "sampling.policies.NonGuidedContextPolicy",
        scale=args.scale,
        diffusion_model=diffusion,
        normalizer=dataset.normalizer,
        preprocess_fns=args.preprocess_fns,
        ## sampling kwargs
        sample_fn=default_sample_fn,
  
        verbose=False,
    )

logger = logger_config()
policy = policy_config()


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
rewards = []
for i in range(10):
    env = dataset.env
    observation = env.reset()

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    frames = []
    conditions = []
    for t in range(500):
        if t==0:
            conditions = np.concatenate([np.array([0]),observation])[None,None,:]
        if t % 10 == 0: print(args.savepath, flush=True)

        ## save state for rendering only
        #state = env.state_vector().copy()
        state = observation

        ## format current observation for conditioning
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

        ## execute action in environment
        print("action: ",action)
        if action > .50:
            action = 1
        else:
            action = 0
        print("rounded: ",action)
        
        next_observation, reward, terminal, _ = env.step(action)
        #frames.append(env.render(mode="rgb_array"))
        new_cond = np.concatenate([np.array([action]),next_observation])
        conditions = np.concatenate([conditions,new_cond[None,None,:]],axis=1)
        ## print reward and score
        total_reward += reward
        #score = env.get_normalized_score(total_reward)
        # print(
        #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        #     f'values: {samples.values} | scale: {args.scale}',
        #     flush=True,
        # )
        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | '
            f'state: {next_observation} | scale: {args.scale}',
            flush=True,
        )
        
        ## update rollout observations
        rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        #logger.log(t, samples, state, rollout)

        if terminal:
            break

        observation = next_observation
    rewards.append(total_reward)
    ## write results to json file at `args.savepath
print("rewards: ",rewards)
print("mean: ",np.mean(rewards))
