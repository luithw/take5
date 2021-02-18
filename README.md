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
from take5.envs.take5_env import Take5Env

sides = 5

env = Take5Env({"sides": sides})
observation = env.reset()
env.render()
for i in range (10):
  action_dict = {}
  for p in range(sides):
    action_dict["p_%i" %p] = i
  observation, reward, done, info = env.step(action_dict)
  env.render()
```

## TODO
 - Support multi-players