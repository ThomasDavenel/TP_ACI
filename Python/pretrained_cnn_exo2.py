# coding=utf-8

import os, sys, signal
from timeit import default_timer as timer

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


######################
# DEFINE THE MODEL
######################

# base pretrained CNN
local_weights_file = '/share/esir2/aci/python/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = VGG16(input_shape=(224, 224,3), include_top=False, weights=local_weights_file)


# Let's make the model non-trainable, since we will only use it for feature extraction; 
# we won't update the weights of the pretrained model during training.
for layer in pre_trained_model.layers:
  layer.trainable = False

# Specify the layer used for feature extraction
model_output = pre_trained_model.get_layer('block5_pool').output
print('last layer output shape:', pre_trained_model.get_layer('block5_pool').output_shape)
print('last layer output shape:', model_output.shape)
# QUESTION 1 : Quelle est la dimension de la sortie choisie ?


# Stick a fully connected classifier on top of last_output
# Flatten the output layer to 1 dimension
x = layers.Flatten()(model_output)
print(x.shape)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)


# QUESTION 2 : Combien de paramètres vont etre appris ? Detaillez le calcul.


# Configure and compile the model
model = Model(pre_trained_model.input, x) # give inputs and output of the model
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['acc'])




######################
# FEED THE MODEL WITH TRAINING DATA
######################

base_dir = "/share/esir2/aci/img_data/cats_and_dogs_sampled/"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# QUESTION 3 : On s'interesse à un sous-echantillon du jeu de donnees "cats and dogs" (disponible sur Kaggle).
# Combien y'a-t-il de classes ?
# Combien y'a-t-il de donnees dans l'ensemble d'apprentissage ? de validation ? de test ?
# Quelle est la distribution des classes ?



# use the flow_from_directory method
# Takes the path to a directory & (optionnally) generates batches of augmented data
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
        train_dir, # This is the source directory for training images
        target_size=(224, 224),  # All images will be resized to 224x224
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 20 using val_datagen generator
validation_generator = datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=20, class_mode='binary')


# Train the model using the features extracted from base model. We'll train on all 3000 images available, for 2 epochs, and validate on all 1,000 validation images.
# validation_steps is the total number of steps (batches of samples) to yield from validation_data generator
# before stopping at the end of every epoch. It should typically be equal to the number of samples of the validation dataset divided by the batch size. 

start = timer()
history = model.fit_generator( # for keras/tensorflow 1.X
      train_generator,
      steps_per_epoch=150,
      epochs=2,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=1)

end = timer()
    
print("Training time: " + str(end - start))

# QUESTION 4 : Quelle est la taille du batch pour l'apprentissage ?

# QUESTION 5 : Combien d'iterations de la descente de gradient sont effectuées durant l'apprentissage ?

# QUESTION 6 : Quelles sont les taux de bonnes classifications obtenus sur l'ensemble d'apprentissage et de validation ?

print("Evaluate the model on test set : \n")
test_generator = datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=1, class_mode='binary')
print(model.metrics_names)
print(model.evaluate_generator(generator=test_generator)) # for tensorflow 1.X


# QUESTION 7 : Quelle est le taux de bonnes classifications obtenu sur l'ensemble de test ?

# QUESTION 8 : A quoi sert ici l'ensemble de validation ?


# matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')



