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
from scipy.optimize import fmin
import numpy as np

from kinematics import forward_kinematics


def from_trans(m):
    """ get x, y, z & angle from transform matrix
    """
    t_x = np.arctan2(m[2, 1], m[2, 2])
    t_y = np.arctan2(-m[2, 0], (m[2, 1] ** 2 + m[2, 2] ** 2) ** .5)
    t_z = np.arctan2(m[1, 0], m[0, 0])
    return np.array([m[0, -1], m[1, -1], m[2, -1], t_x, t_y, t_z])


class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        # YOUR CODE HERE
        init = []
        for joint in self.chains[effector_name]:
            init.append(self.perception.joint[joint])
        if "Arm" in effector_name: # add 2 zeros for translations when it is an Arm
            init.extend((0, 0))
        #init = np.zeros(len(self.chains[effector_name]))
        optimization = fmin(self.error_func, init, args=(effector_name, transform))
        joint_angles = dict(zip(self.chains[effector_name], optimization))

        return joint_angles

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
                if chain == effector_name:
                    self.keyframes[2].append([[self.perception.joint[joint_name], [3, -.01, 0], [3, .01, 0]],
                                              [thetas[joint_name], [3, -.01, 0], [3, .01, 0]]])
                else:
                    self.keyframes[2].append([[self.perception.joint[joint_name], [3, -.01, 0], [3, .01, 0]],
                                              [self.perception.joint[joint_name], [3, -.01, 0], [3, .01, 0]]])

    def error_func(self, init_angles, limb, transform):
        """ error function that uses squared l2 norm
        """
        # this is a version of forward kinematics
        limb_trans = np.eye(4)
        for joint, angle in zip(self.chains_with[limb], list(init_angles)):
            Tl = self.local_trans(joint, angle)
            limb_trans = limb_trans @ Tl

        # error is the squared norm of the angle and position difference
        error = from_trans(transform.T) - from_trans(limb_trans)
        return np.linalg.norm(error) ** 2


if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = np.eye(4)
    #T[-1, 0] = 0.
    T[-1, 1] = .05
    T[-1, 2] = -.26
    print(T)
    agent.set_transforms('LLeg', T)
    agent.run()
