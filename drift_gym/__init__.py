from gym.envs.registration import register

register(
    id='drift-v0',
    entry_point='drift_gym.envs:RealEnv',
)
