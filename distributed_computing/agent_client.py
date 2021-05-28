'''In this file you need to implement remote procedure call (RPC) client

* The agent_server.py has to be implemented first (at least one function is implemented and exported)
* Please implement functions in ClientAgent first, which should request remote call directly
* The PostHandler can be implement in the last step, it provides non-blocking functions, e.g. agent.post.execute_keyframes
 * Hints: [threading](https://docs.python.org/2/library/threading.html) may be needed for monitoring if the task is done
'''

import weakref
import xmlrpc.client
import numpy as np
import marshalling
from threading import Thread
from joint_control.keyframes import leftBackToStand


class PostHandler(object):
    '''the post handler wraps function to be excuted in parallel
    '''
    def __init__(self, obj):
        self.proxy = weakref.proxy(obj)

    def execute_keyframes(self, keyframes):
        '''non-blocking call of ClientAgent.execute_keyframes'''
        # YOUR CODE HERE
        thread = Thread(target=self.proxy.execute_keyframes(keyframes), daemon=True)
        thread.start()

    def set_transform(self, effector_name: str, transform: np.ndarray):
        '''non-blocking call of ClientAgent.set_transform'''
        # YOUR CODE HERE
        packed_transform = marshalling.marshall(transform)
        #print(packed_transform)
        #packed_transform = [1, 2, 3]
        thread = Thread(target=self.proxy.set_transform(effector_name, packed_transform), daemon=True)
        thread.start()

class ClientAgent(object):
    '''ClientAgent request RPC service from remote server
    '''
    # YOUR CODE HERE
    def __init__(self):
        self.post = PostHandler(self)
        self.proxy = xmlrpc.client.ServerProxy('http://localhost:8000/')
    
    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        # YOUR CODE HERE
        print(f"getting angle of {joint_name}")
        return self.proxy.get_angle(joint_name)
    
    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller
        '''
        # YOUR CODE HERE
        print(f"setting angle of {joint_name} to {angle}")
        return self.proxy.set_angle(joint_name, angle)

    def get_posture(self):
        '''return current posture of robot'''
        # YOUR CODE HERE
        print(f"getting posture")
        return self.proxy.get_posture()

    def execute_keyframes(self, keyframes):
        '''excute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        # YOUR CODE HERE
        print("sending keyframes")
        return self.proxy.execute_keyframes(keyframes)

    def get_transform(self, name):
        '''get transform with given name
        '''
        # YOUR CODE HERE
        print("getting transform")
        return marshalling.unmarshall(self.proxy.get_transform(name))

    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        print("setting transform")
        return self.proxy.set_transform(effector_name, marshalling.marshall(transform))

if __name__ == '__main__':
    agent = ClientAgent()
    # TEST CODE HERE
    print(agent.get_angle("RAnkleRoll"))
    print(agent.set_angle("RAnkleRoll", 20))
    print(agent.get_posture())
    print(agent.get_transform("RAnklePitch"))
    print(agent.execute_keyframes(leftBackToStand()))
    A = np.eye(4)
    A[-1, 1] = 0.05
    A[-1, 2] = 0.26
    print(agent.set_transform("LLeg", A))


