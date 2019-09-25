#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 03:43:40 2019

@author: racss
lets create a dueling qnetwort
for gym env
uses ros client and server to replicate a robotic sytem
agent takes an action 
fed to the env client via ros which gives responce
"""
import tensorflow as tf
import numpy as np
import random as rand
import rospy
from collections import deque
from proto_net.srv import EnvFeedBack
from proto_net.msg import env_feed
from dueling_qnet_model import dueling_q_net_model




class ddqnet():
    def __init__(self,state_size,action_size,learning_rate,batch_size,update_rate,epsilon_min,epsilon_decay,sess
):      #instanciates sess and dueling qnet model one as taget the other as the critic
        #instanciates replay memory and sets up epsilon for exploration v exploitation
        self.sess = sess
        self.update_rate=update_rate
        self.state_size=state_size
        self.batch_size=batch_size
        self.memory =deque()
        self.epsilon=1
        self.train_count=0
        self.discount=.99 #Discount Rate
        self.action_size=action_size
        self.epsilon_min=epsilon_min
        self.epsilon_decay=epsilon_decay
        self.actor=dueling_q_net_model(state_size,action_size,learning_rate,"actor")
        self.critic=dueling_q_net_model(state_size,action_size,learning_rate,"critic")
        init = tf.global_variables_initializer()
        sess.run(init)
    def take_action(self,state):
        #uses an epsilon greedy approach to decide which action to take
        if self.epsilon> rand.uniform(0, 1):
            action=rand.randint(0,self.action_size-1)
            return action
        action=self.actor.action_pred.eval(feed_dict={self.actor.state:state}, session=self.sess)[0]
        return action

    def decrease_eps(self):
        #decreases epsilon by given rate until below minimum
        if self.epsilon>self.epsilon_min:
            self.episilon= self.epsilon * self.epsilon_deacy
            
    def remember(self,state,action,state_prime,reward,done):
        #creates replay memory for agent
        self.memory.append((state,action,state_prime,reward,done))
   

    def update_critic(self,current):
        #based on episodes decides when the critct should be updated
        def update_target_graph():
    
            # Get the parameters of our DQNNetwork
            from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "actor")
            
            # Get the parameters of our Target_network
            to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "critic")
        
            op_holder = []
            # Update our target_network parameters with DQNNetwork parameters
            for from_var,to_var in zip(from_vars,to_vars):
                op_holder.append(to_var.assign(from_var))
            return op_holder
                
        if self.train_count%self.update_rate== 0:
            update=update_target_graph()
            print update
            self.sess.run(update)
            

    
    def replay_train(self):
        #randomly samples the agents memory and uses prebuilt model to train
        #uses Q funtion to create the target value 
        #only trains actor critic is held constant until update
        c_state=[]
        actions=[]
        targets=[]
        minibatch = rand.sample(self.memory, self.batch_size)
        for state,action,state_prime,reward,done in minibatch:
            action=self.actor.action_pred.eval(feed_dict={self.actor.state:[state]}, session=self.sess)
            next_state= self.critic.q_vals.eval(feed_dict={self.critic.state:[state_prime]}, session=self.sess)[0]
            if done:
                target = reward
            else:
                target = reward + self.discount * np.amax(next_state)
                
            c_state.append(state)
            actions.append(action)
            targets.append(target)

        c_state=np.reshape(np.asarray(c_state),(self.batch_size,self.state_size))
        actions=np.reshape(np.asarray(actions),(self.batch_size))
        targets=np.reshape(np.asarray(targets),(self.batch_size,1))

        
        self.sess.run(self.actor.updateModel, feed_dict={self.actor.state:c_state, self.actor.target_Q:targets,self.actor.actions:actions})
        self.train_count=self.train_count +1


def env_feedback_client(action, state):
    #instanciates a ros client that gives the server an action and a signal to restart env
    # waits for server
    #converts resulting state from string to array
    rospy.wait_for_service('some_feed_back')
    env_feed_back = rospy.ServiceProxy('some_feed_back', EnvFeedBack,persistent=True)
    resp1 = env_feed_back(action, state)
    #returns state reward and done in propper type
    return np.fromstring(resp1.feed_back.state,dtype=float,sep=','),resp1.feed_back.reward,resp1.feed_back.done

     
def main():
    sess = tf.Session()
    state_size=4
    action_size=2
    learning_rate=.00001
    batch_size=32
    epsilon_min=.01
    update_rate=9
    epsilon_decay=.95
    episodes=20000
    model=ddqnet(state_size,action_size,learning_rate,batch_size,update_rate,epsilon_min,epsilon_decay,sess)
    for i in range(episodes):
        print "reseting env"
        state,reward,done=env_feedback_client(0,"start")#initial state first prev_state also restarts the env
        print "let the games begin"
        while done ==0:
            action=model.take_action(state)
            prev_state=state
            state,reward,done=env_feedback_client(action,"continue")
            model.remember(prev_state,action,state,reward,done)
            print "this was the action:{0} and its reward:{1} done:{2}" .format(action,reward,done)
        if len(model.memory)> model.batch_size:
            print "training and updating critic"
            model.replay_train()
            model.update_critic(i)
        
if __name__ == '__main__':
    main()
    
    
    

