import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.models.temporal import ValueFunctionL2
import numpy as np
import datetime
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')

l2_guide = True

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
if not l2_guide:
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
if l2_guide:
    value_function = ValueFunctionL2()
else:
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

## policies are wrappers around an unconditional diffusion model and a value guide
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

logger = logger_config()
policy = policy_config()


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
reward_list = []
cost_list = []
success_count = 0
num_episodes = 50
for i in range(num_episodes):
    env = dataset.env
    observation = env.reset()

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    total_cost = 0
    frames = []
    terminal = False
    t=0
    while not terminal:

        # if t % 10 == 0: print(args.savepath, flush=True)

        ## save state for rendering only
        #state = env.state_vector().copy()
        state = observation

        ## format current observation for conditioning
        conditions = {0: observation}
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

        ## execute action in environment
        print("action: ",action)
        # action = np.round(action,0).astype(int).item()
        # print("rounded: ",action)

        next_observation, reward, cost, terminal, info = env.step(action)
        #frames.append(env.render(mode="rgb_array"))

        ## print reward and score
        total_reward += reward
        total_cost += cost
        #score = env.get_normalized_score(total_reward)
        # print(
        #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        #     f'values: {samples.values} | scale: {args.scale}',
        #     flush=True,
        # )
        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | Cost: {total_cost:.2f} | '
            f'state: {next_observation} | scale: {args.scale}',
            flush=True,
        )

        ## update rollout observations
        rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        #logger.log(t, samples, state, rollout)
        now = datetime.datetime.now()
        t+=1
        if terminal:
            if info["goal_reached"]:
                success_count += 1
            print("FINAL REWARD: ",total_reward," SUCCESS COUNT: ",success_count)
            env.render(path="/home/fernandi/projects/diffuser/imgs", name=f"planGuided_{now.strftime('%Y-%m-%d_%H:%M:%S')}")
            break

        observation = next_observation
    reward_list.append(total_reward)
    cost_list.append(total_cost)
print("reward list: ",reward_list)
print("average reward: ",np.mean(reward_list))
print("std reward: ",np.std(reward_list))
print("success_rate: ",success_count/num_episodes)
print("average cost: ",np.mean(cost_list))


## write results to json file at `args.savepath`
#logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)