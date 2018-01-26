import keras
from keras.layers import Input, Dense, Flatten, Reshape, Masking
from keras.models import Model
from keras.preprocessing import sequence
import numpy as np
import time

#Input name
inputName = 'cleaned_input_ADP'

#batch size:
batchSize = 10000

#Specify stride:
stride= 1

#Specify input size:
#inputSize = 103
inputSize = 22

#Number of samples
nSamples = 20000

#Epochs
nEpochs = 1000

#Max length - Nearest power of two above inputSize. Automate this.
#maxLen = 128
maxLen = 32

#Bottleneck
bottleneck = 1

#Shall we pad?
pad = True

#Append to output
appendToOut = '_ADP'

#------------------------------#
#Define custom loss to not train on padded values?
#Useless?

#DATA PREPROCESSING
#Load trajectory data
traj = np.loadtxt(inputName)

#Convert data to arrays
x_trj = traj[0:inputSize*nSamples,0:3]
y_trj = traj[inputSize:inputSize*(nSamples+1),0:3]

#Reshape to add number of samples dimension
x_trj = np.reshape(x_trj,(nSamples,inputSize,3))
y_trj = np.reshape(y_trj,(nSamples,inputSize,3))

#Remove geometrical center from each timepoint
x_shifti=np.ones(nSamples)
x_shiftj=np.ones(nSamples)
x_shiftk=np.ones(nSamples)

y_shifti=np.ones(nSamples)
y_shiftj=np.ones(nSamples)
y_shiftk=np.ones(nSamples)

for i in range(0,nSamples):
	x_shifti[i] = np.average(x_trj[i,:,0])
	x_shiftj[i] = np.average(x_trj[i,:,1])
	x_shiftk[i] = np.average(x_trj[i,:,2])

	y_shifti[i] = np.average(y_trj[i,:,0])
	y_shiftj[i] = np.average(y_trj[i,:,1])
	y_shiftk[i] = np.average(y_trj[i,:,2])

	x_trj[i,:,0] = x_trj[i,:,0] - x_shifti[i]
	x_trj[i,:,1] = x_trj[i,:,1] - x_shiftj[i]
	x_trj[i,:,2] = x_trj[i,:,2] - x_shiftk[i] 

	y_trj[i,:,0] = y_trj[i,:,0] - y_shifti[i] 
	y_trj[i,:,1] = y_trj[i,:,1] - y_shiftj[i] 
	y_trj[i,:,2] = y_trj[i,:,2] - y_shiftk[i] 


#Now scale from -1 to 1
#Omit for now.
if(0):
	x_min = np.amin(x_trj)
	y_min = np.amin(y_trj)

	x_trj = x_train - x_min
	y_trj = y_train - y_min

	x_coeff = np.amax(x_trj)
	y_coeff = np.amax(y_trj)

	x_trj = x_trj/x_coeff
	y_trj = y_trj/y_coeff


#Preprocess data - Raise to the nearest power of 2.
#Waste of space to assign to another variable?
if(pad):
	x_train = sequence.pad_sequences(x_trj, maxlen = maxLen, dtype='float', padding = 'post', value=0.)
	y_train = sequence.pad_sequences(y_trj, maxlen = maxLen, dtype='float', padding = 'post', value=0.)
else:
	maxLen = inputSize

#MODEL SETUP
#Setup 1D PCA-like time-lagged autoencoder

input_shape = Input(shape=(maxLen,3))

#Encoder:
inputLayer = Flatten()(input_shape)
encoded = Dense(bottleneck)(inputLayer)

#Decoder:
decoded = Dense(maxLen*3)(encoded)
reshaped = Reshape((maxLen,3))(decoded)

#Mask the padded data.
if(pad):
	decoded = Masking(0.)(decoded)

#Compile model
autoencoder = Model(input_shape,reshaped)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

#Train!
training_start = time.time()
autoencoder.fit(x_train,y_train,epochs=nEpochs, batch_size=batchSize)
training_end = time.time()

#See performance on training data
output = autoencoder.predict(x_train)

#Get the bottleneck values
#Reconstruct the encoder
print(autoencoder.summary())
input_shape2 = Input(shape=(maxLen,3))
inputLayer2 = Flatten()(input_shape2)
encoded2 = Dense(bottleneck,weights=autoencoder.layers[2].get_weights())(inputLayer2)

encoder = Model(input_shape2,encoded2)
encoder.compile(optimizer = 'adamax',loss='mean_squared_error')
print(encoder.summary())

Z = encoder.predict(x_train)
np.savetxt('latent_PCA'+appendToOut,Z)

if(0):
	output = output*y_coeff + y_min

	output[:,0] = output[:,0] + y_shifti
	output[:,1] = output[:,1] + y_shiftj
	output[:,2] = output[:,2] + y_shiftk

	output=np.reshape(output,(1,inputSize*nSamples,3))
	output=np.reshape(output,(inputSize*nSamples,3))

if(0):
	for i in range(0,nSamples):
		x_train[i,:,0] = x_train[i,:,0] + x_shifti[i]
		x_train[i,:,1] = x_train[i,:,1] + x_shiftj[i]
		x_train[i,:,2] = x_train[i,:,2] + x_shiftk[i] 

		y_train[i,:,0] = y_train[i,:,0] + y_shifti[i] 
		y_train[i,:,1] = y_train[i,:,1] + y_shiftj[i] 
		y_train[i,:,2] = y_train[i,:,2] + y_shiftk[i]

#Depad and prepare for output
output=np.reshape(output[:,0:inputSize,:],(1,inputSize*nSamples,3))
output=np.reshape(output,(inputSize*nSamples,3))
y_test=np.reshape(y_train[:,0:inputSize,:],(1,inputSize*nSamples,3))
y_test=np.reshape(y_test,(inputSize*nSamples,3))

np.savetxt('target_trj_PCA'+appendToOut,y_test)
np.savetxt('autoencoded_trj_PCA'+appendToOut,output)

with open('model_details_PCA'+appendToOut,'w') as fh:
	autoencoder.summary(print_fn=lambda x: fh.write(x + '\n'))
	fh.write('#batch size: batchSize = '+str(batchSize)+ '\n')
	fh.write('#Specify stride: stride= '+str(stride)+ ' \n')
	fh.write('#Specify input size: inputSize = '+str(inputSize)+ ' \n')
	fh.write('#Number of samples: nSamples = '+str(nSamples)+ ' \n')
	fh.write('#Epochs: nEpochs = '+str(nEpochs)+ ' \n')
	fh.write('#Max length - Nearest power of two above inputSize. Automate this.: maxLen = '+str(maxLen)+ ' \n')
	fh.write('#Bottleneck: bottleneck = '+str(bottleneck)+ ' \n')
	fh.write('Total training time in seconds: '+str(training_end-training_start)+'\n')
	fh.write('#Shall we pad?: pad = '+str(pad)+'\n')
	fh.write('Total error :' + str(np.average(np.power(output-y_test,2))))


