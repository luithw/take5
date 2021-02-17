import argparse

import gym
import ray
from ray import tune
from ray.tune.registry import register_env

import take5

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="PPO")
    parser.add_argument("--steps", type=int, default=20000000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--restore", type=str, help="Checkpoint dir to restore from")
    args = parser.parse_args()
    env = gym.make('Take5-v0', sides=3)
    register_env('Take5-v0', lambda config: env)
    tune.run(
        args.agent,
        config={
            "env": 'Take5-v0',
            "num_workers": args.workers,
        },
        stop={
            "timesteps_total": args.steps,
        },
        checkpoint_at_end=True,
        checkpoint_freq=10,
        restore=args.restore
    )
