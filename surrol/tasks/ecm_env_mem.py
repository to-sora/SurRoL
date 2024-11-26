import numpy as np
import pybullet as p
from surrol.gym.surrol_env import SurRoLEnv, RENDER_HEIGHT
from surrol.robots.ecm import Ecm
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera
)
from surrol.utils.robotics import get_pose_2d_from_matrix


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

from surrol.tasks.ecm_env import EcmEnv


class EcmEnvMem(EcmEnv):
    """
    Single arm environment using ECM with memory buffer.
    """
    def __init__(self, render_mode: str = None):
        # Memory buffer parameters
        self.k = 5  # Number of past frames to include
        self.c = 2  # Skipping interval
        self.obs_buffer = []  # Buffer to store past observations

        super(EcmEnvMem, self).__init__(render_mode)

    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state()

        if self.has_object:
            pos, _ = get_link_pose(self.obj_id, -1)
            object_pos = np.array(pos)
        else:
            object_pos = np.zeros(0)

        if self.has_object:
            achieved_goal = object_pos.copy()
        else:
            achieved_goal = np.array(get_link_pose(self.ecm.body, self.ecm.EEF_LINK_INDEX)[0])  # eef position

        observation = np.concatenate([
            robot_state, object_pos.ravel(),
        ])

        # Update the observation buffer
        self.obs_buffer.append(observation)
        if len(self.obs_buffer) > self.k * self.c:
            self.obs_buffer.pop(0)

        # Stack observations with zero-padding if necessary
        padded_obs = []
        for i in range(self.k):
            idx = -1 - i * self.c
            if abs(idx) <= len(self.obs_buffer):
                padded_obs.append(self.obs_buffer[idx])
            else:
                # Zero-padding
                padded_obs.append(np.zeros_like(observation))

        # Concatenate the stacked observations
        stacked_obs = np.concatenate(padded_obs, axis=0)
        # print(stacked_obs.shape)  # For debugging; remove or comment out in production

        # Construct the observation dictionary
        obs = {
            'observation': stacked_obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        return obs

    def reset(self):
        # Clear the observation buffer at the start of each episode
        self.obs_buffer = []
        return super(EcmEnvMem, self).reset()
