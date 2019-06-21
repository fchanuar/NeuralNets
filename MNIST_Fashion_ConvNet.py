import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import initializers
from keras import backend as K 
from keras import regularizers
from keras.layers import Input 
from keras.layers import Conv2D, MaxPooling2D

# Load dataset
folder = 'C:\\Users\\BurgerBucks\\Documents\\Proyectos\\NeuralNets\\dataset_mnist\\'

x = np.load(folder+'train_images.npy')
y = np.loadtxt(folder+'train_labels.csv', delimiter=',', skiprows=1)
x_test = np.load(folder+'test_images.npy')

# Split dataset (train/val)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.1)

# Normalization
x_train = x_train.reshape(x_train.shape+(1,))                                    
x_test = x_test.reshape(x_test.shape+(1,))
x_valid = x_valid.reshape(x_valid.shape+(1,))
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_valid= x_valid.astype('float32')/255

# OneHot encode the output
y_train_categorical = to_categorical(y_train)
y_val_categorical = to_categorical(y_valid)

#Plots first dataset's item 
#plt.imshow(x_train[0], cmap='gray_r')
#plt.show()

output_size = 10
# default_initializer = initializers.normal(mean=0, stddev=0.001)
default_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)
K.clear_session()

# Create the neural network
model = Sequential()

# Convoluyional Layers
nerons_per_layer = [32, 64]
for i, neurons in enumerate(nerons_per_layer):
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=x_train.shape[1:],name='Convolucional' + str(i)))

model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Flatten())

# Hidden Layers
nerons_per_layer = [128]
for i, neurons in enumerate(nerons_per_layer):
    model.add(Dense(neurons, activation='relu', kernel_initializer='normal', name='Oculta' + str(i)))
    model.add(Dropout(0.5)) 

# Output Layer
model.add(Dense(output_size, kernel_initializer=default_initializer, name='Salida'))
model.add(Activation('softmax'))

model.summary()

# Compile the net
lr = 0.0001
optim = optimizers.adam(lr=lr)
model.compile(loss = 'categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

# Fit
batch_size = 512
model.fit(x_train, 
          y_train_categorical,
          epochs=20, batch_size=batch_size, 
          verbose=1, 
          validation_data = (x_valid, y_val_categorical),
          )

# Metrics
loss, acc = model.evaluate(x_valid, y_val_categorical)
print(acc)

test_prediction = model.predict(x_test)

test_labels = np.argmax(test_prediction, axis = 1)

