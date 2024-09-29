import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


Data = pd.read_excel("ODE excel.xlsx")
Data = pd.DataFrame(Data)
x = Data['x']
y = Data['y']
a = x.size
b = y.size
x =np.array(x, dtype=np.float32)
y =np.array(y, dtype=np.float32)


x = tf.reshape(x, (a,1))
y = tf.reshape(y, (b,1))
x = tf.Variable(x, dtype=tf.float32)
y_actual = tf.Variable(y, dtype=tf.float32)


def D_y(tape,x,y):
    Dy = tape.gradient(y,x)
    return Dy

def D_yy(tape, x,y):
    Dy = D_y(tape, x, y)
    D_yy = tape.gradient(Dy,x)
    return D_yy

#loss y(0)
def Ic_1():
    return tf.constant([2.78],dtype=tf.float32)

#loss dy(0)
def Ic_2():
    return tf.constant([-0.43],dtype=tf.float32)

epochs = 30000
learning_rate = 0.0001
activation = "tanh"
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
Input_shape, Output_shape = (1,1)

model_1 = tf.keras.models.Sequential()
model_1.add(tf.keras.layers.Input(Input_shape))
model_1.add(tf.keras.layers.Dense(30, activation=activation))
model_1.add(tf.keras.layers.Dense(30, activation=activation))
model_1.add(tf.keras.layers.Dense(30, activation=activation))
model_1.add(tf.keras.layers.Dense(units=Output_shape))


model_1.summary()

epochs = 30000
learning_rate = 0.0001
activation = "tanh"
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
Input_shape, Output_shape = (1,1)

model_2 = tf.keras.models.Sequential()
model_2.add(tf.keras.layers.Input(Input_shape))
model_2.add(tf.keras.layers.Dense(50, activation=activation))
model_2.add(tf.keras.layers.Dense(50, activation=activation))
model_2.add(tf.keras.layers.Dense(50, activation=activation))
model_2.add(tf.keras.layers.Dense(units=Output_shape))


model_2.summary()

epochs = 30000
learning_rate = 0.0001
activation = "tanh"
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
Input_shape, Output_shape = (1,1)

model_3 = tf.keras.models.Sequential()
model_3.add(tf.keras.layers.Input(Input_shape))
model_3.add(tf.keras.layers.Dense(40, activation=activation))
model_3.add(tf.keras.layers.Dense(40, activation=activation))
model_3.add(tf.keras.layers.Dense(40, activation=activation))
model_3.add(tf.keras.layers.Dense(units=Output_shape))


model_3.summary()

Loss_1 = []

for i in range (epochs+1):

    with tf.GradientTape() as tape_c:

        with tf.GradientTape(persistent=True) as tape_b: 

            with tf.GradientTape() as tape_a:

                y = model_1(x,training = True)[:,i:i+1]
                Dy = D_y(tape_a,x,y)

            Dyy = D_yy(tape_b,x,y)
            Loss_Ode_1 = Dyy + 2*Dy + 0.75*y

            Loss_Ode = tf.reduce_mean(tf.square(Loss_Ode_1))
            Loss_Bc = tf.square(y[0]-Ic_1()) + tf.square(Dy[0]-Ic_2())
            Loss_MSE = tf.reduce_mean(tf.square(y_actual-y))
            Total_Loss = Loss_Ode + Loss_Bc + Loss_MSE
            Loss_1.append(Total_Loss)
        if i % 100 == 0:

            print('Epoch: {}\t Total Loss = {}\t dy = {}\t y = {}'.format(i,Total_Loss, Dy[0], y[0]))
            
        model_update = tape_c.gradient(Total_Loss, model_1.trainable_variables)
        optimizer.apply_gradients(zip(model_update, model_1.trainable_variables))

Loss_2 = []

for i in range (epochs+1):

    with tf.GradientTape() as tape_c:

        with tf.GradientTape(persistent=True) as tape_b: 

            with tf.GradientTape() as tape_a:

                y = model_2(x,training = True)[:,i:i+1]
                Dy = D_y(tape_a,x,y)

            Dyy = D_yy(tape_b,x,y)
            Loss_Ode_1 = Dyy + 2*Dy + 0.75*y

            Loss_Ode = tf.reduce_mean(tf.square(Loss_Ode_1))
            Loss_Bc = tf.square(y[0]-Ic_1()) + tf.square(Dy[0]-Ic_2())
            Loss_MSE = tf.reduce_mean(tf.square(y_actual-y))
            Total_Loss = Loss_Ode + Loss_Bc + Loss_MSE
            Loss_2.append(Total_Loss)
        if i % 100 == 0:

            print('Epoch: {}\t Total Loss = {}\t dy = {}\t y = {}'.format(i,Total_Loss, Dy[0], y[0]))
            
        model_update = tape_c.gradient(Total_Loss, model_2.trainable_variables)
        optimizer.apply_gradients(zip(model_update, model_2.trainable_variables))

Loss_3 = []

for i in range (epochs+1):

    with tf.GradientTape() as tape_c:

        with tf.GradientTape(persistent=True) as tape_b: 

            with tf.GradientTape() as tape_a:

                y = model_3(x,training = True)[:,i:i+1]
                Dy = D_y(tape_a,x,y)

            Dyy = D_yy(tape_b,x,y)
            Loss_Ode_1 = Dyy + 2*Dy + 0.75*y

            Loss_Ode = tf.reduce_mean(tf.square(Loss_Ode_1))
            Loss_Bc = tf.square(y[0]-Ic_1()) + tf.square(Dy[0]-Ic_2())
            Loss_MSE = tf.reduce_mean(tf.square(y_actual-y))
            Total_Loss = Loss_Ode + Loss_Bc + Loss_MSE
            Loss_3.append(Total_Loss)
        if i % 100 == 0:

            print('Epoch: {}\t Total Loss = {}\t dy = {}\t y = {}'.format(i,Total_Loss, Dy[0], y[0]))
            
        model_update = tape_c.gradient(Total_Loss, model_3.trainable_variables)
        optimizer.apply_gradients(zip(model_update, model_3.trainable_variables))

from matplotlib import ticker
fig, ax = plt.subplots(figsize = (6,5))
ax.plot(Loss_1, color = 'blue', label = '30 neuron')
ax.plot(Loss_3, color = 'orange',  label = '40 neuron')
ax.plot(Loss_2, color = 'red',  label = '50 neuron')
ax.tick_params(axis = 'x', direction = 'in')
ax.tick_params(axis = 'y', direction = 'in')
ax.tick_params(axis = 'x', which ='minor', direction = 'in')
ax.tick_params(axis = 'y',  which ='minor', direction = 'in')
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1000),)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.00005),)
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')
ax.set_ylim(0,0.0008)
ax.legend(frameon = False, prop = {'size':8})

x_test = np.linspace(-0.1,6,100)
y_test = model_3(x_test)


x_model = tf.range(start=0.1, limit=610, delta=6.1)
x_model = tf.reshape(x_model,(100,1))
x_model = tf.Variable(x_model,dtype=tf.float32)
x_model

import math 
from math import exp

x_actual = []
y_actual = []
for i in range (610):
    x = -0.1 +0.01*i
    x_actual.append(x)

for i in (x_actual):
    b = -0.96*exp(i*-3/2)+3.74*exp(i*-1/2)
    y_actual.append(b)

fig, ax = plt.subplots(figsize = (6,5))
ax.plot(y_actual, color = '#008000', label = 'Actual')
ax.scatter(x_model, y_test, color = 'red', marker = 'x', label = 'PINN')
ax.tick_params(axis = 'x', direction = 'in')
ax.tick_params(axis = 'y', direction = 'in')
ax.legend(frameon = False, prop = {'size':10})



