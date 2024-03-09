import gym
import numpy as np
import carla

class CarlaActions():

    def __init__(self, action_type='continuous'):

        self.action_type = action_type

        if self.action_type == 'carla-original':
            self.discrete_actions = [[0., 0.], [-1.,0.], [-0.5,0.], [-0.25,0.], [0.25,0.], [0.5, 0.], [1.0, 0.], [0., -1.],
                                        [0., -0.5], [0., -0.25], [0., 0.25], [0., 0.5], [0.,1.]]


    def get_action_space(self):

        if self.action_type == 'carla-original':
            return gym.spaces.Discrete(len(self.discrete_actions))

        elif self.action_type == 'continuous':
            low = [0.0, -1.0]
            high = [1.0, 1.0]
            return gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
