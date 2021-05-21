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

from joint_control.keyframes import leftBackToStand, rightBackToStand, leftBellyToStand, hello
from joint_control.pid import PIDAgent
import logging


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

        # if there are no frames to interpolate, then we do nothing
        if keyframes == ([], [], []):
            return target_joints

        # find point in time we need to find the angles of
        if self.start_time < 0:
            self.start_time = perception.time
        elapsed_time = perception.time - self.start_time

        # iterate all joints - compute the angles for each
        for i, (joint_name, joint_times, joint_keys) in enumerate(zip(*keyframes)):
            # min_t = max_t = frame_n = 0
            joint_angles = np.asarray(joint_keys).T[0]

            # if we are at the end, just return the end
            if elapsed_time > joint_times[-1]:
                target_joints[joint_name] = joint_angles[-1]
                if i == len(keyframes[0]) - 1:
                    self.start_time = -1
                    self.keyframes = ([], [], [])
                    logging.info("success: interpolation finished")

            # if we are at the start, just use the first keyframe
            if elapsed_time <= joint_times[0]:
                target_joints[joint_name] = joint_angles[0]

            # if we are in the right window then get right time and then use bezier
            if joint_times[0] < elapsed_time < joint_times[-1]:
                frame_n = np.argmax(np.array(joint_times) > elapsed_time)
                max_t = joint_times[frame_n]
                min_t = joint_times[frame_n-1] if frame_n != 0 else 0
                time = (elapsed_time - min_t) / (max_t - min_t)
                target_joints[joint_name] = self.bezier(joint_keys, frame_n, time)

            # compensate for missing RHipYawPitch
            if "LHipYawPitch" in target_joints:
                target_joints["RHipYawPitch"] = target_joints["LHipYawPitch"]

        return target_joints

    def bezier(self, keys, frame_n, t):
        p0 = keys[frame_n-1][0]
        p3 = keys[frame_n][0]
        p1 = p0 + keys[frame_n-1][2][2]         # check with someone on this, according to docs the 2nd handle value is angle and third is time, but if I use the 2nd it does not work very well
        p2 = p3 + keys[frame_n][1][2]
        return np.power(1 - t, 3) * p0 + 3 * t * np.power(1 - t, 2) * p1 + 3 * np.power(t, 2) * (
                    1 - t) * p2 + np.power(t, 3) * p3

if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = leftBackToStand()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
