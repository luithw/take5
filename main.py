import gym
import numpy as np

import take5

TAKE_LARGEST = False
config = {"sides": 5, "multi_agent": False}
env = gym.make('Take5-v0', config=config)

episodes = 100
returns = []
for e in range(episodes):
    observation = env.reset()
    for i in range(10):
        episode_returns = 0
        if TAKE_LARGEST:
            action = env.hands[0].argmax()
        else:
            action = i
        if config["multi_agent"]:
            if TAKE_LARGEST:
                action = {player: env.hands[p].argmax() for p, player in enumerate(observation.keys())}
            else:
                action = {player: i for player in observation.keys()}
        observation, reward, done, info = env.step(action)
        print(reward)
        if config["multi_agent"]:
            episode_returns += list(reward.values())[0]
        else:
            episode_returns += reward
        env.render()
    returns.append(episode_returns)

returns = np.array(returns)
print("Multi-agent: %r, take largest card policy: %r, Reward max: %f, mean: %f, min: %f" % (
    config["multi_agent"], TAKE_LARGEST, returns.max(), returns.mean(), returns.min()))
