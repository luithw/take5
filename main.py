import gym
import take5

config = {"sides": 5, "multi_agent": False}
env = gym.make('Take5-v0', config=config)
observation = env.reset()
env.render()
for i in range (10):
  action = i
  if config["multi_agent"]:
    action = {player: action for player in observation.keys()}
  observation, reward, done, info = env.step(action)
  env.render()
