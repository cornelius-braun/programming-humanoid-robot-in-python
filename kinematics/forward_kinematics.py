'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    the local_trans has to consider different joint axes and link parameters for different joints
'''

# add PYTHONPATH
import os
import sys
import numpy as np
from joint_control.angle_interpolation import AngleInterpolationAgent

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))


class ForwardKinematicsAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: np.eye(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
                       # YOUR CODE HERE
                       'LArm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll'],
                       'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll'],
                       'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
                       'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
                       }
        self.joint_lengths = {'HeadYaw': (0., 0., .1265), 'HeadPitch': (0., 0., 0.),
                              'LShoulderPitch': (0., .098, .1), 'LShoulderRoll': (0., 0., 0.),
                              'LElbowYaw': (.105, .015, 0.), 'LElbowRoll': (0., 0., 0.),
                              'RShoulderPitch': (0., .098, .1), 'RShoulderRoll': (0., 0., 0.),
                              'RElbowYaw': (.105, .015, 0.), 'RElbowRoll': (0., 0., 0.),
                              'LHipYawPitch': (0., .05, -.085), 'RHipYawPitch': (0., .05, -.085),
                              'LHipRoll': (0., 0., 0.), 'LHipPitch': (0., 0., 0.), 'LKneePitch': (0., 0., -.1),
                              'LAnklePitch': (0., 0., -.1029), 'LAnkleRoll': (0., 0., 0.),
                              'RHipRoll': (0., 0., 0.), 'RHipPitch': (0., 0., 0.), 'RKneePitch': (0., 0., -.1),
                              'RAnklePitch': (0., 0., -.1029), 'RAnkleRoll': (0., 0., 0.),
                              }

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''
        T = np.eye(4)
        # YOUR CODE HERE
        s = np.sin(joint_angle)
        c = np.cos(joint_angle)
        trans = self.joint_lengths[joint_name]

        rot_x = np.array([[1, 0, 0, trans[0]],
                          [0, c, -s, trans[1]],
                          [0, s, c, trans[2]],
                          [0, 0, 0, 1]])
        rot_y = np.array([[c, 0, s, trans[0]],
                          [0, 1, 0, trans[1]],
                          [-s, 0, c, trans[2]],
                          [0, 0, 0, 1]])
        rot_z = np.array([[c, s, 0, trans[0]],
                          [-s, c, 0, trans[1]],
                          [0, 0, 1, trans[2]],
                          [0, 0, 0, 1]])

        if 'Roll' in joint_name:
            T = T @ rot_x
        if 'Pitch' in joint_name:
            T = T @ rot_y
        if 'Yaw' in joint_name:
            T = T @ rot_z
        return T

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain_joints in self.chains.values():
            T = np.eye(4)
            for joint in chain_joints:
                angle = joints[joint]
                Tl = self.local_trans(joint, angle)
                # YOUR CODE HERE
                T = T @ Tl
                self.transforms[joint] = T


if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()
