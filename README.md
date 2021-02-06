# Take 5 Card Game OpenAI Gym Environment

This is a simple imeplemntation of the classic take 5 card game.

Currently it only supports single player and the other players uses the "Largest Card" policy.

## Installation
In your virtual environment, navigate to this folder and install the package.
```
pip install -e .
```

Make this location available in your Python path.
```
export PYTHONPATH=$PYTHONPATH:/path/to/take5
```

## Hello World
```
import gym
import take5

env = gym.make('Take5-v0', sides=3)
observation = env.reset()
env.render()
for i in range (10):
  observation, reward, done, info = env.step(i)
  env.render()
```

## TODO
 - Support multi-players