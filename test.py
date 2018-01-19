import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
import numpy as np

#Specify filter window size:
filterWindow = 10

#Specify stride:
stride= 1

#Number of filters:
nFilters = 30

#Specify input size:
inputSize = 100

#Number of samples
nSamples = 10

#Epochs
nEpochs = 2000

#------------------------------#


# Load trajectory data
traj = np.loadtxt('cleaned.gro')

#Testing - load a single timepoint
x_train = traj[0:inputSize*nSamples,0:3]
y_train = traj[100:inputSize*nSamples+100,0:3]

#Save for evaluation of autoencoder performance
y_test = y_train

input_shape = Input(shape=(inputSize,3))

#Setup 1D Convolutional Autoencoder
#First, encoder:
x = Conv1D(nFilters,filterWindow, strides=stride, activation='relu',padding='same')(input_shape)
x = MaxPooling1D(2)(x)
x = Conv1D(nFilters,filterWindow, strides=stride, activation='relu',padding='same')(x)
x = MaxPooling1D(2)(x)

#Now, decoder:
x = Conv1D(nFilters,filterWindow, strides=stride, activation='relu',padding='same')(x)
x= UpSampling1D(2)(x)
x = Conv1D(nFilters,filterWindow, strides=stride, activation='relu',padding='same')(x)
x= UpSampling1D(2)(x)
decoded = Conv1D(3,filterWindow, strides=stride, activation='sigmoid',padding='same')(x)

autoencoder = Model(input_shape,decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

print(autoencoder.summary())


#Reshape to add number of samples dimension
x_train = np.reshape(x_train,(nSamples,inputSize,3))
y_train = np.reshape(y_train,(nSamples,inputSize,3))


#Remove geometrical center
x_shifti = np.average(x_train[:,0])
x_shiftj = np.average(x_train[:,1])
x_shiftk = np.average(x_train[:,2])

y_shifti = np.average(y_train[:,0])
y_shiftj = np.average(y_train[:,1])
y_shiftk = np.average(y_train[:,2])

x_train[:,0] = x_train[:,0] - x_shifti
x_train[:,1] = x_train[:,1] - x_shiftj 
x_train[:,2] = x_train[:,2] - x_shiftk 

y_train[:,0] = y_train[:,0] - y_shifti 
y_train[:,0] = y_train[:,1] - y_shiftj 
y_train[:,0] = y_train[:,2] - y_shiftk 


#Now scale from 0 to 1
x_min = np.amin(x_train)
y_min = np.amin(y_train)

x_train = x_train - x_min
y_train = y_train - y_min

x_coeff = np.amax(x_train)
y_coeff = np.amax(y_train)

x_train = x_train/x_coeff
y_train = y_train/y_coeff

print x_train

#Train!
autoencoder.fit(x_train,y_train,epochs=nEpochs, batch_size=10)

#See performance on training data
output = autoencoder.predict(x_train)
output = output*y_coeff + y_min

output[:,0] = output[:,0] + y_shifti
output[:,1] = output[:,1] + y_shiftj
output[:,2] = output[:,2] + y_shiftk

output=np.reshape(output,(1,inputSize*nSamples,3))
output=np.reshape(output,(inputSize*nSamples,3))
np.savetxt('output',output)




