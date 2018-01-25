import keras
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
from keras.preprocessing import sequence
import numpy as np
import time

#batch size:
batchSize = 100

#Specify filter window size:
filterWindow = 10

#Specify stride:
stride= 1

#Number of filters:
nFilters = 100

#Specify input size:
inputSize = 103

#Number of samples
nSamples = 1000

#Epochs
nEpochs = 10000

#Max length - Nearest power of two above inputSize. Automate this.
maxLen = 103

#Bottleneck
bottleneck = 2

#------------------------------#


#DATA PREPROCESSING
# Load trajectory data
traj = np.loadtxt('cleaned_input')

#Testing - load a single timepoint
x_train = traj[0:inputSize*nSamples,0:3]
y_train = traj[inputSize:inputSize*(nSamples+1),0:3]

#Reshape to add number of samples dimension
x_train = np.reshape(x_train,(nSamples,inputSize,3))
y_train = np.reshape(y_train,(nSamples,inputSize,3))

#Remove geometrical center from each timepoint
x_shifti=np.ones(nSamples)
x_shiftj=np.ones(nSamples)
x_shiftk=np.ones(nSamples)

y_shifti=np.ones(nSamples)
y_shiftj=np.ones(nSamples)
y_shiftk=np.ones(nSamples)

for i in range(0,nSamples):
	x_shifti[i] = np.average(x_train[i,:,0])
	x_shiftj[i] = np.average(x_train[i,:,1])
	x_shiftk[i] = np.average(x_train[i,:,2])

	y_shifti[i] = np.average(y_train[i,:,0])
	y_shiftj[i] = np.average(y_train[i,:,1])
	y_shiftk[i] = np.average(y_train[i,:,2])

	x_train[i,:,0] = x_train[i,:,0] - x_shifti[i]
	x_train[i,:,1] = x_train[i,:,1] - x_shiftj[i]
	x_train[i,:,2] = x_train[i,:,2] - x_shiftk[i] 

	y_train[i,:,0] = y_train[i,:,0] - y_shifti[i] 
	y_train[i,:,1] = y_train[i,:,1] - y_shiftj[i] 
	y_train[i,:,2] = y_train[i,:,2] - y_shiftk[i] 

#print(y_train)

#Now scale from -1 to 1
#Omit for now.
if(0):
	x_min = np.amin(x_train)
	y_min = np.amin(y_train)

	x_train = x_train - x_min
	y_train = y_train - y_min

	x_coeff = np.amax(x_train)
	y_coeff = np.amax(y_train)

	x_train = x_train/x_coeff
	y_train = y_train/y_coeff

#print(x_train[0,:,:])

#Preprocess data - Raise to the nearest power of 2.
#Not currently needed.
if(0):
	x_train = sequence.pad_sequences(x_train, maxlen = maxLen, dtype='float', padding = 'post', value=0.)
	y_train = sequence.pad_sequences(y_train, maxlen = maxLen, dtype='float', padding = 'post', value=0.)

#print(x_train[0,:,:])
#exit()

#Save for evaluation of autoencoder performance
y_test = y_train

input_shape = Input(shape=(maxLen,3))

#MODEL SETUP
#Setup 1D PCA-like time-lagged autoencoder

#Encoder:
inputLayer = Flatten()(input_shape)
encoded = Dense(bottleneck)(inputLayer)

#Decoder:
decoded = Dense(maxLen*3)(encoded)
reshaped = Reshape((maxLen,3))(decoded)

autoencoder = Model(input_shape,reshaped)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')


#Train!
training_start = time.time()
autoencoder.fit(x_train,y_train,epochs=nEpochs, batch_size=batchSize)
training_end = time.time()

#See performance on training data
output = autoencoder.predict(x_train)

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

output=np.reshape(output,(1,maxLen*nSamples,3))
output=np.reshape(output,(maxLen*nSamples,3))
y_test=np.reshape(y_train,(1,maxLen*nSamples,3))
y_test=np.reshape(y_test,(maxLen*nSamples,3))

np.savetxt('target_trj_PCA',y_test)
np.savetxt('autoencoded_trj_PCA',output)
with open('model_details_single','w') as fh:
	autoencoder.summary(print_fn=lambda x: fh.write(x + '\n'))
	fh.write('#batch size: batchSize = '+str(batchSize)+ '\n')
	fh.write('#Specify filter window size: filterWindow = '+str(filterWindow)+ '\n')
	fh.write('#Specify stride: stride= '+str(stride)+ ' \n')
	fh.write('#Number of filters: nFilters = '+str(nFilters)+ ' \n')
	fh.write('#Specify input size: inputSize = '+str(inputSize)+ ' \n')
	fh.write('#Number of samples: nSamples = '+str(nSamples)+ ' \n')
	fh.write('#Epochs: nEpochs = '+str(nEpochs)+ ' \n')
	fh.write('#Max length - Nearest power of two above inputSize. Automate this.: maxLen = '+str(maxLen)+ ' \n')
	fh.write('#Bottleneck: bottleneck = '+str(bottleneck)+ ' \n')
	fh.write('Total training time in seconds: '+str(training_end-training_start)+'\n')
	fh.write('Total error :' + str(np.average(np.power(output-y_test,2))))


