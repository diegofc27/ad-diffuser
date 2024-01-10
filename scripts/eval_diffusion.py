import diffuser.sampling as sampling
import diffuser.utils as utils
import numpy as np
import diffuser
import datetime 
from diffuser.environments.safe_grid import Safe_Grid, Safe_Grid_v1, Safe_Grid_Simple, Safe_Grid_v2


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
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#

    env = Safe_Grid_v2(max_number_steps=50)
    observation = env.reset()
    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    total_cost = 0
    frames = []

    for t in range(50):

        if t % 10 == 0: print(args.savepath, flush=True)

        ## save state for rendering only
        #state = env.state_vector().copy()
        state = observation
        

        ## format current observation for conditioning
        
        conditions = {0: observation}
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=True)

        ## execute action in environment
        #action = np.round(action,0).astype(int).item()
        next_observation, reward,cost, terminal, info = env.step(action)
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
            f'state: {next_observation}',
            flush=True,
        )

        ## update rollout observations
        rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        #logger.log(t, samples, state, rollout)
        #get time
        now = datetime.datetime.now()

        if terminal:
            env.render(path="/home/fernandi/projects/diffuser/imgs", name=f"safe_grid_v2_{now.strftime('%Y-%m-%d_%H:%M:%S')}")
            env.close()
            if info["goal_reached"]:
                print("****Goal reached!****")
            break

        observation = next_observation

    return total_reward, total_cost
