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


def load_dataset():
    """
    Load Dataset from file
    """
    folder = 'C:\\Users\\BurgerBucks\\Documents\\Proyectos\\varios\\dataset_mnist\\'

    x = np.load(folder+'train_images.npy')
    y = np.loadtxt(folder+'train_labels.csv', delimiter=',', skiprows=1)
    x_test = np.load(folder+'test_images.npy')


def split_dataset(ptest_size = 0.1):
    """
    Split Dataset in train y test
    """
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = ptest_size)


def create_model(nerons_per_layer, activ_func):
    model = Sequential()

    #Input Layer
    model.add(Flatten(input_shape=x_train.shape[1:]),name='Entrada')

    #Hidden Layers
    for i, neurons in enumerate(nerons_per_layer):
        model.add(Dense(neurons, activation=activ_func, kernel_initializer='normal', name='Oculta'+i))

    #Output Layer
    model.add(Dense(output_size, kernel_initializer=default_initializer, name='Salida'))
    model.add(Activation('softmax'))

    return model


#Load dataset
load_dataset

#Split dataset (train/test)
split_dataset

#OneHot encode the output
y_train_categorical = to_categorical(y_train)
y_val_categorical = to_categorical(y_valid)

#Plots first dataset's item 
plt.imshow(x_train[0], cmap='gray_r')

output_size = 10
#default_initializer = initializers.normal(mean=0, stddev=0.001)
default_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)
K.clear_session()


#Create the neural network
model=create_model ([200,100,60,30,3],'relu')
model.summary()

# Compilo el modelo
lr = 0.001
#optim = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
optim = optimizers.adam(lr=lr)
model.compile(loss = 'categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

batch_size = 512
model.fit(x_train, 
          y_train_categorical,
          epochs=40, batch_size=batch_size, 
          verbose=1, 
          validation_data = (x_valid, y_val_categorical),
          )

loss, acc = model.evaluate(x_valid, y_val_categorical)

print(acc)

test_prediction = model.predict(x_test)

test_labels = np.argmax(test_prediction, axis = 1)

