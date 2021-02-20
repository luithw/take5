import gym
import numpy as np

import take5

TAKE_LARGEST = False
config = {"sides": 5, "multi_agent": False}
env = gym.make('Take5-v0', config=config)


def largest_policy(observation):
    hand_hl = observation[20:30][0]
    hand = observation[20:30][2]

    card_idx = np.argmax(hand_hl)
    to_select_card_idx = np.argmax(hand)
    jumps = to_select_card_idx - card_idx
    if jumps < 0:
        action = 0
    elif jumps > 0:
        action = 1
    else:
        # current card is the highest so select it
        action = 2

    return action


episodes = 1
returns = []
for e in range(episodes):
    observation = env.reset()
    is_done = False
    while not is_done:
        episode_returns = 0

        if config["multi_agent"]:
            if TAKE_LARGEST:
                action = {player: largest_policy(observation[player]) for player in observation.keys()}
            else:
                action = {player: env.action_space.sample() for player in observation.keys()}
        else:
            if TAKE_LARGEST:
                action = largest_policy(observation)
            else:
                action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        if config["multi_agent"]:
            is_done = done["__all__"]
        else:
            is_done = done

        if config["multi_agent"]:
            episode_returns += list(reward.values())[0]
        else:
            episode_returns += reward

        env.render()
        print("Reward: ", reward)
    returns.append(episode_returns)

returns = np.array(returns)
print("Multi-agent: %r, take largest card policy: %r, Reward max: %f, mean: %f, min: %f" % (
    config["multi_agent"], TAKE_LARGEST, returns.max(), returns.mean(), returns.min()))
