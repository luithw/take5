import gym
import pytest
import numpy as np

import take5


@pytest.mark.parametrize(
    'multi_agent, sides',
    [
        (True, 3),
        (False, 3),
        (True, 5),
        (False, 5),
        (False, 3),
        (True, 3)
    ]
)
def test_playing(multi_agent, sides):
    env = gym.make('Take5-v0', config={"multi_agent": multi_agent,
                                       "sides": sides})
    observation = env.reset()
    env.render()
    is_done = False
    while not is_done:
        action = 0
        if multi_agent:
            action = {player: action for player in observation.keys()}
        observation, reward, done, info = env.step(action)

        if multi_agent:
            is_done = done["__all__"]
        else:
            is_done = done

        if multi_agent:
            assert type(observation) == dict
            assert type(reward) == dict
            assert type(done) == dict
            assert type(info) == dict
            assert len(observation.keys()) == sides
            assert len(reward.keys()) == sides
            assert "__all__" in done.keys()
            assert len(info.keys()) == sides
        else:
            assert type(observation) == np.ndarray or observation is None
            assert type(reward) == np.float32
            assert type(done) == bool
            assert type(info) == dict


if __name__ == '__main__':
    pytest.main([__file__])
