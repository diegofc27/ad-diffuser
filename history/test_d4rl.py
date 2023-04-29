import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import diffuser.utils as utils

# # Create the environment
# env = gym.make('maze2d-umaze-v1')

# # d4rl abides by the OpenAI gym interface
# env.reset()
# env.step(env.action_space.sample())

# # Each task is associated with a dataset
# # dataset contains observations, actions, rewards, terminals, and infos
# dataset = env.get_dataset()
# print(dataset['observations'].shape) # An N x dim_observation Numpy array of observations
# print("episodes")
# print((dataset['terminals']==True).sum())
dataset_config = utils.Config(
    'datasets.SequenceDataset',
    savepath=('/home/fernandi/projects/diffuser/logs/example', 'dataset_config.pkl'),
    env="CartPole-v1",
    horizon=32,
    normalizer="GaussianNormalizer",
    preprocess_fns=[],
    use_padding=True,
    max_path_length=1000,
)
dataset = dataset_config()
