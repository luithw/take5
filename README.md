# Take 5 Card Game OpenAI Gym Environment

This is a simple implementation of the classic [take 5](https://www.amigo.games/game/take) card game.

Currently it only supports single player and the other players uses the "Largest Card" policy.

## Installation
In your virtual environment, navigate to this folder and install the package.
```
pip install -e .
pip install -r requirements.txt
```

Make this location available in your Python path.
```
export PYTHONPATH=$PYTHONPATH:/path/to/take5
```

## Hello World
```
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
```

## Test Test Test
```
pytest
```

## TODO
 - Support multi-players