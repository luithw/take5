import gym
import take5


env = gym.make('Take5-v0')
observation = env.reset()
observation, reward, done, info = env.step(2)
