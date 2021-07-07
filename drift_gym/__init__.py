from gym.envs.registration import register

register(
    id='Real-v0',
    entry_point='drift_gym.envs:RealEnv',
)
