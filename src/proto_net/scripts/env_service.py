#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 02:50:21 2019

@author: racss
creating a publisher that publishes once
"""
        
 
import rospy
import gym
import numpy as np
from proto_net.msg import env_feed
from proto_net.srv import *
class enviroment():
    def __init__(self,enviroment_name):
        self.env=gym.make(enviroment_name)
        self.done=False  
        self.step=0
    def reset(self):
        state = self.env.reset()
        self.prev_state = np.reshape(state, [1, state.shape[0]])
        return(self.prev_state)
     
        
    def responce_conveter(self,state,done,reward):
        #takes output of the env and converts it to propper ROS msg
        #converts to std_msg
        if done == True:
            done =1
        else: 
            done=0
        state=np.array2string(state, precision=5, separator=',',suppress_small=True)
        state=state.strip("[]")
        return state,done,reward
            
                
    def perform_n_see(self,action,state):
        #performs the action from client and uses env
        if self.step ==0 or state=="start":  #checks for initial and reset conditions
            state=self.reset()
            state=np.array2string(state, precision=5, separator=',',suppress_small=True)
            state=state.strip("[]")
            reward=0
            done=0
            self.step=1
            return state,reward,done
        
        next_state, reward, done,_=self.env.step(action)
        state,conv_done,reward=self.responce_conveter(next_state,done, reward ) # converts result to porpper ros std msg types
        if conv_done == 1: #sets up enviroment for reset
           self.step=0

        return state,reward,done


def handle_env_feed_back(req):
    #properly gives service required return parameter
    msg=env_feed()
    state,reward,done=env.perform_n_see(req.action,req.state)
#    print(done)
    msg.state=state
    msg.reward=reward
    msg.done=done
    return msg

def enviroment_feedback_server():
    #instantiates the ros server for your service
    rospy.init_node('env_feedback_server')
    s = rospy.Service('some_feed_back', EnvFeedBack, handle_env_feed_back)
    print "starting service Envhadler, consider your enviroment hadled."
    rospy.spin()
    
    
    
if __name__ == '__main__':
    env=enviroment("CartPole-v1")
    enviroment_feedback_server()


