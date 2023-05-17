import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from diffuser.datasets.d4rl import load_environment

# Parallel environments
#env = make_vec_env("CartPole-v1", n_envs=4)
env = load_environment('halfcheetah-medium-expert-v2')

model = PPO("MlpPolicy", env, device="cuda:2", verbose=)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole",device="cuda:2")

obs = env.reset()
total_reward = 0
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards
    print(total_reward)
    #env.render()