from time import sleep
from collections import namedtuple

import numpy as np
import gym
from gym import spaces
from mpu9250_jmdev.registers import (
                        AK8963_ADDRESS,
                        MPU9050_ADDRESS_68,
                        GFS_1000,
                        AFS_8G,
                        AK8963_BIT_16,
                        AK8963_MODE_C100HZ
                    )
from mpu9250_jmdev.mpu_9250 import MPU9250

from jetracer.nvidia_racecar import NvidiaRacecar

Observation = namedtuple('Observation', ['accel_x', 'accel_y', 'accel_z',
                         'gyro_x', 'gyro_y', 'gyro_z'])

THROTTLE_GAIN = -1


class RealEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self, throttle=0.003, dt=0.05, horizon=200):
        self.default_throttle = throttle
        self.dt = dt
        self.horizon = horizon

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(6,))

        self.car = NvidiaRacecar()

        self.mpu = MPU9250(
            address_ak=AK8963_ADDRESS,
            address_mpu_master=MPU9050_ADDRESS_68,  # In 0x68 Address
            address_mpu_slave=None,
            bus=0,  # TODO set that in config file
            gfs=GFS_1000,
            afs=AFS_8G,
            mfs=AK8963_BIT_16,
            mode=AK8963_MODE_C100HZ)

        self.mpu.configure()

    def step(self, action):
        self.steps_taken += 1
        observation = self.get_observation()
        reward = self.get_reward(observation)

        self.car.steering = action
        sleep(self.dt)

        if self.steps_taken > self.horizon:
            done = True
        else:
            done = False

        info = {}

        return observation, reward, done, info

    def get_reward(self, observation):
        return np.abs(observation.gyro_z)

    def get_observation(self):
        accel_x, accel_y, accel_z = self.mpu.readAccelerometerMaster()
        gyro_x, gyro_y, gyro_z = self.mpu.readGyroscopeMaster()
        # magneto_x, magneto_y, magneto_z = mpu.readMagnetometerMaster()

        return Observation(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)

    def reset(self):
        self.steps_taken = 0
        self.car.throttle = self.default_throttle*THROTTLE_GAIN
        self.car.steering = 0

        return self.get_observation()

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        self.car.throttle = -0.5*THROTTLE_GAIN  # brake
