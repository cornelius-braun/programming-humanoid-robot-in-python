'''In this file you need to implement remote procedure call (RPC) server

* There are different RPC libraries for python, such as xmlrpclib, json-rpc. You are free to choose.
* The following functions have to be implemented and exported:
 * get_angle
 * set_angle
 * get_posture
 * execute_keyframes
 * get_transform
 * set_transform
* You can test RPC server with ipython before implementing agent_client.py
'''

# add PYTHONPATH
import os
import sys
import marshalling
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'kinematics'))
from kinematics.inverse_kinematics import InverseKinematicsAgent
from xmlrpc.server import SimpleXMLRPCServer
from socketserver import ThreadingMixIn
from threading import Thread

# Restrict to a particular path.
#class RequestHandler(SimpleXMLRPCRequestHandler):
#    rpc_paths = ('/robocup',)

class SimpleThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


class ServerAgent(InverseKinematicsAgent):
    '''ServerAgent provides RPC service
    '''
    # YOUR CODE HERE
    def __init__(self, port=8000):
        super(ServerAgent, self).__init__()
        self.server = SimpleThreadedXMLRPCServer(("localhost", port), allow_none=True)
        # self.server.RequestHandlerClass.rpc_paths = tuple('/robocup')
        self.server.register_introspection_functions()
        self.server.register_multicall_functions()
        self.server.register_instance(self)
        self.thread = Thread(target=self.server.serve_forever(), daemon=True)
        self.thread.start()
        print(f"listening on localhost : port {port}")

    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        # YOUR CODE HERE
        print(f"getting angle for {joint_name}")
        return self.perception.joint[joint_name]
    
    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller
        '''
        # YOUR CODE HERE
        print(f"setting angle {angle} for {joint_name}")
        if joint_name in self.target_joints.keys():
            self.target_joints[joint_name] = angle
            return True
        return False

    def get_posture(self):
        '''return current posture of robot'''
        # YOUR CODE HERE
        print("getting posture")
        # TODO: how can I inherit from the PostureRecognitionAgent??
        return self.posture
        # return False

    def execute_keyframes(self, keyframes):
        '''excute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        # YOUR CODE HERE
        # TODO: wait until done to return true by using a done flag
        self.start_time = -1
        self.keyframes = keyframes
        while self.start_time != -1:
            pass
        return True

    def get_transform(self, name):
        '''get transform with given name
        '''
        # YOUR CODE HERE
        print(f"getting transform of {name}")
        return marshalling.marshall(self.transforms[name])

    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        print(f"setting transform of {effector_name}")
        self.set_transforms(effector_name, marshalling.unmarshall(transform))
        self.execute_keyframes(keyframes=self.keyframes)
        return True

if __name__ == '__main__':
    agent = ServerAgent()
    agent.run()

