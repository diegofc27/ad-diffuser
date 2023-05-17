import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import numpy as np
import diffuser

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

env = dataset.env
observation = env.reset()

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
frames = []

for t in range(550):

    if t % 10 == 0: print(args.savepath, flush=True)

    ## save state for rendering only
    #state = env.state_vector().copy()
    state = observation

    ## format current observation for conditioning
    conditions = {0: observation}
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

    ## execute action in environment
    print("action: ",action)
    action = np.round(action,0).astype(int).item()
    print("rounded: ",action)

    next_observation, reward, terminal, _ = env.step(action)
    #frames.append(env.render(mode="rgb_array"))

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

## write results to json file at `args.savepath`
