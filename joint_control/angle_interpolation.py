'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''
import numpy as np

from pid import PIDAgent
from keyframes import hello

class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.start_time = -1

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}
        # YOUR CODE HERE

        if self.start_time < 0:
            self.start_time = self.perception.time

        elapsed_time = perception.time - self.start_time    # why interval here?
        n_skipped = 0

        # this returns the first value of frame as natural condition of interpolation
        if elapsed_time <= 0:
            target_joints = dict(zip(keyframes[0], [frame[0][0] for frame in keyframes[2]]))
            return target_joints

        for joint_name, joint_times, joint_keys in zip(*keyframes):

            # find first keyframe after current time

            # either the frame we look for is before the window
            if elapsed_time < joint_times[0]:
                next_keyframe = joint_keys[0]
                # last keyframe = pseudo keyframe for nothing happened
                pre_keyframe = [perception.joint[joint_name], [3, 0, 0], [3, 0, 0]]
                pre_time = 0
                next_time = joint_times[0]

            # or were at the end
            elif elapsed_time >= joint_times[- 1]:
                pre_keyframe = next_keyframe = joint_keys[-1]
                pre_time = next_time = joint_times[-1]

                # mark keyframes as done
                # self.keyframe_done = True

            # base case: we are in the middle
            else:
                joint_times_np = np.asarray(joint_times)
                idx = (np.abs(joint_times_np - elapsed_time)).argmin()
                pre_keyframe = joint_keys[idx - 1]
                next_keyframe = joint_keys[idx]
                pre_time = joint_times[idx - 1]
                next_time = joint_times[idx]

            # create the time-angle-pairs
            p0 = (pre_time, pre_keyframe[0])
            p1 = (pre_time + pre_keyframe[2][1], pre_keyframe[0] + pre_keyframe[2][2])
            p2 = (next_time + next_keyframe[1][1], next_keyframe[0] + next_keyframe[1][2])
            p3 = (next_time, next_keyframe[0])

            # get the time
            i = 1 if pre_time == next_time else (elapsed_time-pre_time) / (next_time-pre_time)
            t = self.getcubic(p0, p1, p2, p3, elapsed_time)

            # add joint after evaluation
            target_joints[joint_name] = self.bezier(p0[1], p1[1], p2[1], p3[1], t)

        return target_joints

    def bezier(self, p0, p1, p2, p3, t):
        return np.power(1-t,3) * p0 + 3 * np.power((1 - t), 2) * t * p1 \
               + 3 * (t - 1) * np.power(t, 2) * p2 + np.power(t, 3) * p3

if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
