from gym.envs.registration import register

register(
    id='Take5-v0',
    entry_point='take5.envs.take5_env:Take5Env',
)
