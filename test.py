import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
import numpy as np

#Specify filter window size:
filterWindow = 3

#Specify stride:
stride= 3

#Number of filters:
nFilters = 11

#Specify input size:
inputSize = 103


#------------------------------#


# Load trajectory data
traj = np.loadtxt('cleaned.gro')

#Testing
x_train = traj[0:103,0:3].T
y_train = traj[103:206,0:3].T

input_shape = Input(shape=(103,3))

#Setup 1D Convolutional Autoencoder
#First, encoder:
model = Conv1D(nFilters,filterWindow, strides=stride, activation='relu')(input_shape)
encoded = MaxPooling1D(2)(model)

#Now, decoder:
model = Conv1D(nFilters,filterWindow, strides=stride, activation='relu')(encoded)
model = UpSampling1D(2)(model)
decoded = Conv1D(1,filterWindow, strides=stride, activation='sigmoid')(model)


model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

x_train = np.reshape(x_train,(1,103,3))
y_train = np.reshape(y_train,(1,103,3))

model.fit(x_train, y_train, epochs=20, batch_size=128)
#score = model.evaluate(x_test, y_test, batch_size=128)




