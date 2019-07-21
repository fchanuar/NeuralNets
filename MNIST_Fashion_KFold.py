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
from keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold

# Load dataset
folder = 'C:\\Users\\BurgerBucks\\Documents\\Proyectos\\NeuralNets\\dataset_mnist\\'

x = np.load(folder+'train_images.npy')
y = np.loadtxt(folder+'train_labels.csv', delimiter=',', skiprows=1)
x_test = np.load(folder+'test_images.npy')

# Split dataset (train/val)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.1)

# Normalization
x_test = x_test.astype('float32')/255
x = x.astype('float32')/255

# OneHot encode the output
y_categorical = to_categorical (y)

#Plots first dataset's item 
#plt.imshow(x_train[0], cmap='gray_r')
#plt.show()

#Modelo para KFold
output_size = y_categorical.shape[1]
default_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)
#K.clear_session()

def create_model():
    #Creacion de la red
    model = Sequential()
    
    #Input Layer
    model.add(Flatten(input_shape=x.shape[1:],name='Entrada'))
    
    #Hidden Layers
    nerons_per_layer = [700, 300, 60, 30, 10]
    for i, neurons in enumerate(nerons_per_layer):
        model.add(Dense(neurons, kernel_initializer='normal', name='Oculta_' + str(i+1),use_bias=False))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))
        model.add(Activation('relu'))
    
    #Output Layer
    model.add(Dense(output_size, kernel_initializer=default_initializer, name='Salida'))
    model.add(Activation('softmax'))
    
    lr = 0.001
    optim = optimizers.adam(lr=lr)
   
    model.compile(loss = 'categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    
    return model

#K-Fold
num_folds = 5
batch_size = 256
epocs = 100
skf = StratifiedKFold(n_splits=num_folds, random_state=42)
print(skf)
kfold_generator = skf.split(x, y)

#Se entrena cada fold por separado y se guarda el mejor juego de parámetros
for fold, (train_indices, valid_indices) in enumerate(kfold_generator):
    x_train = x[train_indices]
    y_train = y_categorical[train_indices]
    x_valid = x[valid_indices]
    y_valid = y_categorical[valid_indices]
    
    # Callbacks
    ## Callback para guardar pesos
    checkpointer = ModelCheckpoint(filepath=f'checkpoint.fold-{fold}.hdf5', verbose=1,
                                   save_best_only=True, monitor='val_acc', mode='max')
    earlyStopping=EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=80, verbose=1, restore_best_weights=True)
    reduceLR=ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=5,
                               verbose=1,
                               min_delta=0,
                               cooldown=0, min_lr=0)
    model = create_model()   #Se crea el modelo aquí porque es necesario que cada fold arranque desde 0 (sino en la proxima epoca tiene los pesos entrenados de la anterior
    
    model.fit(x_train, 
        y_train,
        epochs=epocs, batch_size=batch_size, 
        verbose=1, 
        validation_data = (x_valid, y_valid),
        callbacks=[checkpointer, earlyStopping, reduceLR],
    )
    
    model.load_weights(f'checkpoint.fold-{fold}.hdf5')
    
    # Metrics
    loss, acc = model.evaluate(x_valid, y_valid)
    print(f'Fold {fold} -> ACC: {acc}\n\n')


model = create_model()
final_prediction = 0
for fold in range(num_folds):
    model.load_weights(f'checkpoint.fold-{fold}.hdf5')
    final_prediction = final_prediction + model.predict(x_test)
    
test_labels = np.argmax(final_prediction, axis = 1)

model.predict(x_test)

print(np.argmax(final_prediction))