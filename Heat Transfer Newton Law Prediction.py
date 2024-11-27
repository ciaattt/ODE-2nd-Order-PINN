
import tensorflow as tf
import scipy as scipy
import sklearn as sklearn
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class NewtonHeat_ODE:

    def __init__ (self,x,y_true):

        self.y_true = y_true
        self.x = x

        self.weight_init = tf.keras.initializers.he_normal()
        self.learning_rate = 0.0001

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.activation = tf.keras.activations.tanh

        self.epochs = 20000
        self.y_model = self.Neural_net(self.x,
                                       self.y_true,
                                       self.weight_init,
                                       self.activation,
                                       self.epochs,
                                       self.optimizer)

    def Loss_data (self,T, y_true, dT):

        Loss_data = tf.reduce_mean(tf.square(T-y_true))
        Loss_Boundary = tf.reduce_mean(tf.square(T[0]-y_true[0]))
        PDE_Loss = tf.reduce_mean(tf.square(dT + 0.76*(T - 5)))
        Total_Loss = Loss_data + Loss_Boundary + PDE_Loss
        return Total_Loss

    def Neural_net (self,x,y_true,weight_init,activation,epochs,optimizer):
        
        input = tf.keras.Input(shape=(1,))
        layers_1 = tf.keras.layers.Dense(30, activation, weight_init)(input)
        layers_2 = tf.keras.layers.Dense(30, activation, weight_init)(layers_1)
        layers_3 = tf.keras.layers.Dense(30, activation, weight_init)(layers_2)
        output = tf.keras.layers.Dense (1)(layers_3)
        model = tf.keras.Model(input,output)
        Y = []

        for i in range (epochs):
            
            with tf.GradientTape() as NN:

                with tf.GradientTape() as Ad:
                    T = model(x, training = True)
                    dT = Ad.gradient(T,x)

                data_loss = self.Loss_data(T,y_true,dT)
                model_update = NN.gradient(data_loss,model.trainable_variables)
                optimizer.apply_gradients(zip(model_update,model.trainable_variables))
            
            if i%100 ==0:
                print (data_loss)
                
        
model = (NewtonHeat_ODE(x,y_true).y_model)

        
x = data['x']
y = data['y']
x = np.array(x,dtype=np.float32)
y = np.array(y,dtype=np.float32)
x = tf.Variable(tf.constant(x, shape=(len(x),1)),dtype=tf.float32)   
y_true = tf.Variable(tf.constant(y, shape=(len(y),1)),dtype=tf.float32) 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
