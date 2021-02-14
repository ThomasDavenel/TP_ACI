# coding=utf-8

#########################################
### Extract features from pretrained cnn
#########################################
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

from timeit import default_timer as timer
import os, sys
import numpy as np


base_dir = "C:/Users/thoma/OneDrive/Bureau/Ecole/ESIR2/Semestre8/ACI/TP/CatsAndDog/cats_and_dogs_sampled/"
output_dir="./"
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

valid_cats_dir = os.path.join(valid_dir, 'cats')
valid_dogs_dir = os.path.join(valid_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')


local_weights_file = 'C:/Users/thoma/OneDrive/Bureau/Ecole/ESIR2/Semestre8/ACI/TP/python/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
model_vgg16 = VGG16(include_top=False, weights=local_weights_file, pooling='max')


# QUESTION : (a) Quelle est la dimension des descripteurs extraits ici ? 
# (b) A quelle couche du reseau correspondent-ils ?
# (c) Decrivez sous quelle forme sont retournees les donnees d'apprentissage (attributs + classe) ?


def extract_features(class_input_dir, class_id):

    class_vgg16_feature_list = []
    im_nb=0

    for fname in os.listdir(class_input_dir):
        print (str(im_nb+1) + " Compute descriptors for " + fname)
        img = image.load_img(os.path.join(class_input_dir, fname), target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        vgg16_feature = model_vgg16.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        class_vgg16_feature_list.append(vgg16_feature_np.flatten())

        im_nb=im_nb+1


    class_vgg16_feature_list_np = np.array(class_vgg16_feature_list)
    y_train=np.full(im_nb,class_id)

    return class_vgg16_feature_list_np, y_train



if __name__ == "__main__":
    model_vgg16.summary()
    
    start1 = timer()
    """
    cat_vgg16_feature_list_np, cat_y_train = extract_features(train_cats_dir,0)
    dog_vgg16_feature_list_np, dog_y_train = extract_features(train_dogs_dir,1)

    vgg16_feature_list_np=np.concatenate((cat_vgg16_feature_list_np, dog_vgg16_feature_list_np))
    y_train=np.concatenate((cat_y_train,dog_y_train))

    np.save(output_dir + "vgg16_train_descriptors.npy", vgg16_feature_list_np)
    np.save(output_dir + "vgg16_train_target.npy", y_train)
    """
    
    """
    cat_vgg16_feature_list_np, cat_y_test = extract_features(test_cats_dir,0)
    dog_vgg16_feature_list_np, dog_y_test = extract_features(test_dogs_dir,1)

    vgg16_feature_list_np=np.concatenate((cat_vgg16_feature_list_np, dog_vgg16_feature_list_np))
    y_test=np.concatenate((cat_y_test,dog_y_test))

    np.save(output_dir + "vgg16_test_descriptors.npy", vgg16_feature_list_np)
    """
    np.save(output_dir + "vgg16_test_target.npy", y_test)
    end1 = timer()

    print("Train extraction Time: " + str(end1 - start1))
 
    sys.exit(0)

