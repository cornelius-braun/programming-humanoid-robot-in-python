'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''

from forward_kinematics import ForwardKinematicsAgent
# import autograd.numpy as np
# from autograd import grad
# import jax.numpy as jax
# from jax import grad, jit
from scipy.optimize import fmin
import numpy as np


class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        # YOUR CODE HERE
        optimization = fmin(self.error_func, np.zeros(len(self.chains[effector_name])), args=(effector_name, transform))
        joint_angles = dict(zip(self.chains[effector_name], optimization))

        return joint_angles

    def fwd_kin_2(self, chain_joints):
        T = np.eye(4)
        for joint, angle in chain_joints.items():
            Tl = self.local_trans(joint, angle)
            # YOUR CODE HERE
            T = T @ Tl
            # self.transforms[joint] = T
        return T

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        self.keyframes = ([], [], [])  # the result joint angles have to fill in
        thetas = self.inverse_kinematics(effector_name, transform)

        # loop all joints and append the angles that we just computed or 0 on other effectors
        for chain in self.chains:
            for joint_name in self.chains[chain]:
                self.keyframes[0].append(joint_name)
                self.keyframes[1].append([2., 6.])
                print(chain, effector_name)
                if chain == effector_name:
                    self.keyframes[2].append([[self.perception.joint[joint_name], [3, -1, 0], [3, 1, 0]],
                                              [thetas[joint_name], [3, -1, 0], [3, 1, 0]]])
                else:
                    self.keyframes[2].append([[self.perception.joint[joint_name], [3, -1, 0], [3, 1, 0]],
                                              [self.perception.joint[joint_name], [3, -1, 0], [3, 1, 0]]])

    def from_trans(self, m):
        """ get x, y, z & angle from transform matrix
        """
        t = 0
        if m[0, 0] == 1:
            t = np.arctan2(m[2, 1], m[1, 1])
        elif m[1, 1] == 1:
            t = np.arctan2(m[0, 2], m[0, 0])
        elif m[2, 2] == 1:
            t = np.arctan2(m[1, 0], m[0, 0])
        return np.array([m[0, -1], m[1, -1], m[2, -1], t])

    def error_func(self, angles, limb, transform):
        """ error function that uses squared l2 norm
        """
        limb_angles = dict(zip(self.chains[limb], list(angles)))
        limb_trans = self.fwd_kin_2(limb_angles)
        error = self.from_trans(transform) - self.from_trans(limb_trans)
        return error @ error


if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = np.eye(4)
    T[-1, 1] = 0.23
    T[-1, 2] = 0.26
    T[-1, 3] = .5
    agent.set_transforms('LLeg', T)
    agent.run()
