import gym
import pytest
import numpy as np

import take5


@pytest.mark.parametrize(
    'multi_agent, sides, play_illegal',
    [
     (True, 3, False),
     (False, 3, False),
     (True, 5, False),
     (False, 5, False),
     (False, 3, True),
     (True, 3, True)
    ]
)
def test_playing(multi_agent, sides, play_illegal):
  config = {"multi_agent": multi_agent,
            "sides": sides,
            "illegal_moves_limit": 3}
  env = gym.make('Take5-v0', config=config)
  observation = env.reset()
  env.render()
  for i in range (10):
    action = 0 if play_illegal else i
    if multi_agent:
      action = {player: action for player in observation.keys()}
    observation, reward, done, info = env.step(action)
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
      assert type(observation) == np.ndarray
      assert type(reward) == np.float64
      assert type(done) == bool
      assert type(info) == dict
      if play_illegal and i == config["illegal_moves_limit"] + 1:
        assert done


if __name__ == '__main__':
    pytest.main([__file__])
