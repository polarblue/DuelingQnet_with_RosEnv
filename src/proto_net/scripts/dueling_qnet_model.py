#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 03:43:40 2019

@author: racss
lets create a dueling qqn model
takes env size and action space size to create model
"""


import tensorflow as tf

class dueling_q_net_model():
    def __init__(self,state_size,action_size,learning_rate,type_net):
        #instanciates wieghts for actor and critct and sets up graph 
        #sets up learning rate
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.state=tf.placeholder(shape=[None,state_size],dtype=tf.float32)
        self.target_Q = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,self.action_size,dtype=tf.float32)
        

        with tf.variable_scope(type_net):
            self.weights = {    
                'wd1' : tf.get_variable('W0', shape=(state_size,100), initializer=tf.contrib.layers.xavier_initializer()),
                'wd2' : tf.get_variable('W1', shape=(100,50), initializer=tf.contrib.layers.xavier_initializer()),
                's_d_w': tf.get_variable('W2', shape=(50,25), initializer=tf.contrib.layers.xavier_initializer()), 
                's_w':tf.get_variable('W3', shape=(25,1), initializer=tf.contrib.layers.xavier_initializer()), 
                's_a_d_w':tf.get_variable('W4', shape=(50,25), initializer=tf.contrib.layers.xavier_initializer()), 
                's_a_w': tf.get_variable('W5', shape=(25,action_size), initializer=tf.contrib.layers.xavier_initializer()), 
                    }
            
            self.bias = {

                'bd1': tf.get_variable('B0', shape=(100), initializer=tf.contrib.layers.xavier_initializer()),
                'bd2': tf.get_variable('B1', shape=(50), initializer=tf.contrib.layers.xavier_initializer()),
                's_d_b': tf.get_variable('B2', shape=(25), initializer=tf.contrib.layers.xavier_initializer()),
                's_b': tf.get_variable('B3', shape=(1), initializer=tf.contrib.layers.xavier_initializer()),
                's_a_d_b': tf.get_variable('B4', shape=(25), initializer=tf.contrib.layers.xavier_initializer()),
                's_a_b': tf.get_variable('B5', shape=(action_size), initializer=tf.contrib.layers.xavier_initializer()),
                    }
    
        self.q_vals=self.dueling_q_net(self.state)
        self.action_pred=tf.argmax(self.q_vals,1)

        error = tf.square(self.target_Q -(tf.reduce_sum(tf.multiply(self.q_vals, self.actions_onehot), axis=1)))
        self.loss = tf.reduce_mean(error)
        trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.updateModel = trainer.minimize(self.loss)
        
#    def conv2d(self,x, W, b, strides=1):
#        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
#        x = tf.nn.bias_add(x, b)
#        return tf.nn.relu(x) 
    
        
    
#    def flat_dense(self,x, W ,b):
#        x = tf.contrib.slim.flatten(x)
#        x=tf.matmul(x,W)
#        x=tf.nn.bias_add(x,b)
#        return tf.nn.relu(x)

    def dense(self,x, W ,b):
        x=tf.matmul(x,W)
        x=tf.nn.bias_add(x,b)
        return tf.nn.relu(x)

    
    def s_value(self,x,w,b):
        #takes last layer and produces a stream just for value of a state
         value = tf.matmul(x,w)
         value=tf.nn.bias_add(value,b)
         return value
 
    def s_a_value(self,x,w,b):
        #takes previus later and produces a stream for the advantage 
        #of action at given state
         value = tf.matmul(x,w)
         value=tf.nn.bias_add(value,b)
         return value  
     
    def aggregate(self,state_values,advantage_values):
        # creates a q value separated by state value and advantage
        Q_val = state_values + tf.subtract(advantage_values,tf.reduce_mean(advantage_values,axis=1,keep_dims=True))
        return Q_val
    
    
    def set_weights(self,model_weights,model_bias):
        self.weights=model_weights
        self.bias=model_bias
        
    def get_weights(self):
        #creates method for getting weight required 
        return self.weights,self.bias
        
    def dueling_q_net(self,state):
        #creates model graph
        dens1=self.dense(state,self.weights['wd1'],self.bias['bd1'])
        dens2=self.dense(dens1,self.weights['wd2'],self.bias['bd2'])
        s_v_dense=self.dense(dens2,self.weights['s_d_w'],self.bias['s_d_b'])
        s_a_v_dense=self.dense(dens2,self.weights['s_a_d_w'],self.bias['s_a_d_b'])
        state_values=self.s_value(s_v_dense,self.weights['s_w'],self.bias['s_b'])
        advantage_values=self.s_a_value(s_a_v_dense,self.weights['s_a_w'],self.bias['s_a_b'])
        q_val=self.aggregate(state_values,advantage_values)
        return q_val
    



        

        

