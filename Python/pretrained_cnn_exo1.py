# coding=utf-8

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers

import numpy as np
import os, sys


##########################################################
# Analyse d'un modèle entrainé sur Imagenet
##########################################################

# Chargement d'un modele entraine sur Imagenet
model_full = VGG16(weights='imagenet')


# QUESTION 1 : Analyser le modele : model_full.summary()
# combien de couches de convolution ? de pooling ? de couches entierement connectées ? 
# Quelle est la taille de la couche d'entree ? de sortie ? pourquoi ?
# Combien y'a-t-il de parametres appris ?
# Représentez graphiquement le modèle.


# Forward propagation : on fait passer une image dans le reseau
img_path = '/share/esir2/aci/img_data/cats_and_dogs_sampled/train/cats/cat.83.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)

# our image is now represented by a NumPy array of shape (3, 224, 224),
# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
# pass it through the network. This is related to batch learning (first dimension = batch size)

img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data) # Adequate the image to the format the model requires (for ex. subtracting the mean RGB pixel intensity from the ImageNet dataset)

# Get the output
output = model_full.predict(img_data)
print("last fully connected size : ", output.shape)
# convert the probabilities to class labels
label = decode_predictions(output)
# print the top5 classification results
for i in range (5):
	print('%s (%.2f%%)' % (label[0][i][1], label[0][i][2]*100))


# QUESTION 2 : Quelle est la classe prédite de l'image ?
# Reessayer avec une autre image de chat. Quel est le résultat ?
# Reessayer avec une image de chien. Quel est le résultat ?
# Est-ce pertinent ?



#####################################################################
# Utilisation d'un modele pre-entrainé pour extraire des descripteurs
# 		-> Acces aux couches intermediaires
#####################################################################

# By specifying the include_top=False argument, we load a network that doesn't include the classification layers at the top
# —> ideal for feature extraction.

# Here we already have locally stored the weights
local_weights_file = '/share/esir2/aci/python/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = VGG16(input_shape=(224, 224, 3),include_top=False, weights=None)
pre_trained_model.load_weights(local_weights_file)
#pre_trained_model.summary()



# QUESTION 3 : Quelle est la dimension de la sortie de ce réseau lorsque l'on met une image en entrée ?
# Comment s'appelle la couche qu'on récupère dans l'architecture du réseau ?
output = pre_trained_model.predict(img_data)
print("last convolutionnal size : ",output.shape)



modelfeatures1 = VGG16(include_top=False, weights=local_weights_file, pooling='avg')
output_features1 = modelfeatures1.predict(img_data)

# QUESTION 4 : Quelle est la dimension de output_features1 ? Que vient-on de faire ?
print("output_features1 : ",output_features1.shape)




modelfeatures2 = VGG16(weights=local_weights_file, include_top=False, pooling='max')
output_features2 = modelfeatures2.predict(img_data)
print("output_features2 : ",output_features2.shape)

# QUESTION 5 :
# La dimension de 'output_features1' est-elle identique à 'output_features2' ? 
# Les valeurs de 'output_features1' sont-elles identiques à 'output_features2' ? pourquoi ?


# On souhaite maintenant extraire une couche intermediaire : la couche 'block4_pool'
model_extractfeatures = Model(inputs=pre_trained_model.input, outputs=pre_trained_model.get_layer('block4_pool').output)
block_04 = model_extractfeatures.predict(img_data)
# QUESTION 6 : Quelle est la dimension de la sortie obtenue ?
print("block_04: ",block_04.shape)


# For tensorflow 2.X
#feature = layers.GlobalMaxPooling2D()(block_04)
#print("feature: ",feature.shape)

# For tensorflow 1.X
# Some conversions are needed
import tensorflow as tf
data_tf = tf.convert_to_tensor(block_04, np.float32) # need to convert numpy array to tensor for Pooling
feature = layers.GlobalMaxPooling2D()(data_tf)

from tensorflow.keras import backend as K
feature=K.eval(feature) # converting back tensor to numpy array
print("feature: ",feature.shape)





