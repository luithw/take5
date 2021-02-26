import argparse

import ray
from ray import tune

from take5.envs.take5_env import Take5Env

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="APEX")
    parser.add_argument("--steps", type=int, default=320_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--restore", type=str, help="Checkpoint dir to restore from")
    parser.add_argument("--self_play", type=bool, default=False)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    tune.run(
        args.agent,
        config={
            "env": Take5Env,
            "num_workers": args.workers,
            "env_config": {
                "sides": 3,
                "multi_agent": args.self_play
            },
            "num_envs_per_worker": 8,
            "lr": 5e-5,
            "gamma": 0.99,
        },
        stop={
            "timesteps_total": args.steps,
        },
        checkpoint_at_end=True,
        checkpoint_freq=100_000,
        restore=args.restore,
        name=args.name
    )
