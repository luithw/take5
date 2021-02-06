import gym
import take5

env = gym.make('Take5-v0', sides=5, debug=True)
observation = env.reset()
env.render()
for i in range (10):
  observation, reward, done, info = env.step(i)
  env.render()
