install.packages("tensorflow")
install.packages("keras")
library(tensorflow)
library(keras)
source("./fonctions_utiles.R")

################################################################################
#              Exo 1 : Effet de la fonction d'activation 
#            et des itérations de la descente de gradient
################################################################################

dataset = read.table("./data/exo1_keras.txt", header = T)
plot(dataset[which(dataset$y==0),1:2], col="red", xlim = c(-3,4), ylim = c(-5,9))
points(dataset[which(dataset$y==1),1:2], col="blue")

# Préparation des données pour utiliser des réseaux de neurones
# On met les individus et leur description dans une matrice, 
# et les classes dans un vecteur à part 
train_x = data.matrix(dataset[,1:2])
train_y = dataset$y


# Définition d'un réseau de neurone avec 1 couche cachée de 10 neurones

model1 <- keras_model_sequential()
model1 %>% 
  layer_dense(units =10, input_shape =2) %>%  # pour la première couche, il faut indiquer le nombre d'input 
  # dans input_shape (dépend de la dimension des données d'entrée), et le nombre de neurones de la couche cachée dans units. 
  # On choisit d'abord une activation linéaire sur cette couche (par défaut, en l'absence de specifications)
  layer_dense(units = 1, activation = "sigmoid") # couche de sortie : 1 neurone dont la valeur de sortie représente la proba 
# d'être de la classe 1 (activation sigmoid, toujours)

# On compile le modèle: toujours la même commande, sauf lorsqu'il y a plus de 2 classes (cf. exo 4)
model1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

# On lance l'apprentissage avec la commande fit. Paramètres : 
# - la matrice contenant les descriptions des individus
# - un vecteur contenant les classes des individus
# - epochs représente le nombre de fois où l'algorithme d'apprentissage voit tous les exemples de l'ensemble d'apprentissage
# - batch représente le nb d'individus dans le batch d'apprentissage (lié à la descente de gradient, on ne modifie pas pour l'instant)
model1 %>% fit(train_x, train_y, epochs = 100, batch.size = 10) 

###### Question 1: pour 100 epochs et un mini-batch de taille 10, quel est le nombre d'itération de la descente de gradient ?
# (i.e. le nombre de fois où les poids ont été modifiés ?)


# Vous devez voir apparaître deux courbes qui représentent en fonction des "epochs" :
# 1) l'évolution du "loss" (fonction que minimise le réseau de neurone, ici la binary cross-entropy) sur l'ensemble d'apprentissage
# 2) l'évolution du taux de bonne classification (accuracy) sur l'ensemble d'apprentissage

###### Question 2: 
# Comment se comporte le loss ? et l'accuracy ? Quelles sont leurs valeurs finales ?

# Affichage de la frontière (même fonctionnement que les autres fonctions similaires)
dessiner_frontiere_NN(train_x, train_y, model1, -4,4,-8,8, col = c("red", "blue"))

###### Question 3:  Quelle est la forme de la frontière ? Est-elle adaptée ici ?


# On ajoute la commande activation = "relu" après input_shape = 2 dans la définition de la première couche. 
###### Question 4: qu'est-ce que cela signifie ? quelle est la conséquence attendue ?

model2 <- keras_model_sequential()
model2 %>%
  layer_dense(units = 10, input_shape =2, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") # couche de sortie : 1 valeur qui représente la proba 

model2 %>% compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = c('accuracy'))
model2 %>% fit(train_x, train_y, epochs = 100, batch.size = 10)
dessiner_frontiere_NN(train_x, train_y, model2, -4,4,-8,8, col = c("red", "blue"))

###### Question 5: La forme de la frontière a t'elle changée ? Pensez-vous qu'elle puisse être améliorée ? Comment ?
# Les valeurs finales de loss et d'accuracy ont elles changées ?


### On peut poursuivre l'apprentissage du modèle en relancant la commande fit (sans réinitialiser le modèle). La descente de gradient 
# recommencera là oû elle s'était arretée auparavant.
# L'option view_metrics = F permet de désactiver l'affichage en temps réel de l'évolution des métriques,
# afin d'accélerer l'apprentissage
model2 %>% fit(train_x, train_y, epochs = 100, batch.size = 10, view_metrics = F) 
# On refait 100 itérations en partant du modèle déjà appris auparavant.

###### Question 6: L'accuracy s'est-elle améliorée avec ces 100 nouvelles itérations ?
# Regardez la frontière

dessiner_frontiere_NN(train_x, train_y, model2, -4,4,-8,8, col = c("red", "blue"))

### Ajouter encore des itérations, jusqu'à atteindre une accuracy de 1 (vous devriez avoir besoin d'environ
# 300 ou 400 itérations supplémentaires, mais allez y 100 par 100)

model2 %>% fit(train_x, train_y, epochs = 400, batch.size = 10, view_metrics = F) 

###### Question 7: Affichez la frontière. Vous convient-elle ?


dessiner_frontiere_NN(train_x, train_y, model2, -4,4,-8,8, col = c("red", "blue"))


###############################################################
#     Exo 2 : Structure à plusieurs couches cachées
##############################################################

dataset = read.table("./data/exo2_keras.txt", header = T)
plot(dataset[which(dataset$y==0),1:2], col="red", xlim = c(0,20), ylim = c(0,21))
points(dataset[which(dataset$y==1),1:2], col="blue")

# Préparation des données pour utiliser des réseaux de neurones.
# On met les individus et leur description dans une matrice, 
# et les classes dans un vecteur à part 
train_x = data.matrix(dataset[,1:2])
train_y = dataset$y


#### On va maintenant utiliser un réseau comprenant 2 couches cachées, 
# dont 1 couche avec une activation relu et une autre avec une activation sigmoid:
model <- keras_model_sequential()
model %>%
  layer_dense(units = 20, input_shape =2, activation = "sigmoid") %>% # première couche cachée : 20 neurones, sigmoid 
  layer_dense(units = 20, activation = "relu")%>% # deuxième couche cachée : 20 neurones, relu
  layer_dense(units = 1, activation = "sigmoid") # couche de sortie : 1 neurone car 2 classes, sigmoid

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

model %>% fit(train_x, train_y, epochs = 50, batch.size = 10, view_metrics = F) 

###### Question 1: Quelle est l'accuracy après 50 epochs ?

# Affichez la frontière :
dessiner_frontiere_NN(train_x, train_y, model, 0,20,0,21, col = c("red", "blue"))

###### Question 2: Combien de points sont mal classés ?


### Continuez l'apprentissage en ajoutant des itérations jusqu'à avoir une accuracy d'au moins 0.9868
model %>% fit(train_x, train_y, epochs = 100, batch.size = 10, view_metrics = F)
# Puis affichez la frontière.
###### Question 3: Quel phénomène voyez-vous apparaître ?

dessiner_frontiere_NN(train_x, train_y, model, 0,20,0,21, col = c("red", "blue"))



############################################################
#   Exo 3 : Apprentissage d'un réseau de neurone en pratique
############################################################

# Pour éviter le sur-apprentissage dû à un trop grand nombre d'itérations, il faut passer par une séparation
# en ensembles d'apprentissage/validation et test. L'ensemble de validation sert ici à estimer l'erreur réelle
# au cours des itérations de la descente de gradient afin d'arrêter l'apprentissage avant le sur-apprentissage.
# La librairie keras effectue elle-même la séparation de l'ensemble des données utilisées en apprentissage 
# en ensemble d'apprentissage et de validation. Cet exercice illustre ce processus.

ex3=read.table("./data/exo3_keras.txt", header = T)
head(ex3)
table(ex3$Classe)
plot(ex3[which(ex3$Classe==0),1:2], col="red", xlim = c(-1.2,1.2), ylim = c(-1.2,1.2))
points(ex3[which(ex3$Classe==1),1:2], col="blue")

# On va séparer les données en 2 ensembles seulement : apprentissage qui va contenir 80% des données et test 20%
# C'est keras qui va s'occuper de créer un ensemble de validation à l'intérieur de celui d'apprentissage quand
# il fera l'apprentissage (fit)
nall = nrow(ex3) #total number of rows in data
ntrain = floor(0.80 * nall) # number of rows for train: 80%
ntest = floor(0.20* nall) # number of rows for test: 20%
index = sample(nall) # permutation aléatoire des nombres 1, 2, 3 , ... nall

train_x = ex3[index[1:ntrain],1:2] # ensemble d'apprentisssage
train_y = ex3[index[1:ntrain],3] # labels d'apprentissage

test_x = ex3[index[(ntrain+1):nall],1:2] # ensemble de test
test_y = ex3[index[(ntrain+1):nall],3] # labels de test

train_x = matrix(unlist(train_x), ncol = 2)
test_x = matrix(unlist(test_x), ncol = 2)

# on met en place un réseau à 3 couches cachées:
model <- keras_model_sequential()
model %>%
  layer_dense(units = 80, input_shape =2, activation = 'relu') %>%
  layer_dense(units = 40, activation = 'relu') %>%
  layer_dense(units = 30, activation = 'relu') %>%
  layer_dense(units = 1,activation = 'sigmoid') 

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

###### Question 1: Représentez graphiquement ce réseau.
# En utilisant la commande, summary(model), déterminez le nombre de paramètres qui doivent être appris pour ce modèle.

print(summary(model))


# Dans la commande fit, il y a le paramètre 'validation_split' qui permet de créer un ensemble de validation 
# à partir des exemples réservés pour l'apprentissage.
# Il suffit d'indiquer la proportion d'éléments que l'on souhaite mettre dans l'ensemble de validation.
# On choisit 0.2 (20%) ici.
###### Question 2: Quelle est la taille de l'ensemble d'apprentissage ? de l'ensemble de validation ?


# Apprentissage du modèle
model %>% fit(train_x, train_y, epochs = 500, batch.size = 10, validation_split = 0.2)

tmp =nrow(train_x)
print(tmp)
print(tmp*0.2)
# Il y a maintenant deux courbes de loss : celle d'apprentissage et celle de validation. De même pour l'accuracy
###### Question 3: Qu'observe t'on sur la courbe de loss validation ? Interprétez ce résultat.



# On peut demander à la fonction 'fit' de keras de s'arrêter lorsque l'on estime que l'on a atteint suffisamment
# d'itérations (avant d'arriver dans la zone de surapprentissage)
# Pour cela, on lui demande de s'arrêter lorsque le loss en validation a cessé de diminuer depuis un certain temps.

# Réinitialiser d'abord le modèle:
model <- keras_model_sequential()
model %>%
  layer_dense(units = 80, input_shape =2, activation = 'relu') %>%
  layer_dense(units = 40, activation = 'relu') %>%
  layer_dense(units = 30, activation = 'relu') %>%
  layer_dense(units = 1,activation = 'sigmoid') 

model %>% compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = c('accuracy'))

# Puis on va refaire le fit en demandant de s'arreter (early_stopping) 
# lorsque le loss en validation (val_los) n'a plus diminué lors des 100 dernières epochs (patience = 100).
# Le modèle retenu est celui qui a obtenu le dernier minimum de val_loss avant l'arrêt.
model %>% fit(train_x, train_y, epochs = 500, batch.size = 10, validation_split = 0.2,callbacks = list(callback_early_stopping("val_loss", patience = 100)))

###### Question 4: combien d'epochs ont été réalisées ?




#### Prédiction sur l'ensemble de test :
# Affichage de la frontiere et des donnees de test (en noir)
tmp_x = test_x[1:5,1:2]
plot(tmp_x[,1:2], col="black", xlim = c(-1.2,1.2), ylim = c(-1.2,1.2))
dessiner_frontiere_NN(train_x, train_y, model, -1.2,1.2,-1.2,1.2, col = c("red", "blue"))

# La commande predict permet d'obtenir la sortie du réseau de neurones, c'est à dire la probabilité d'être de la classe 1 (car on
# est dans le cas d'une classification binaire 0 ou 1):
tmp=predict(model, test_x)
nb = nrow(tmp)
tmp
var=0
for(i in 1:nb){
  if(floor(tmp[i]*2)==test_y[i]) var=var+1
}
var/24
# Vous obtenez les probabilités d etre de la classe 1 pour les 24 individus de test
# Pour transformer ces probas en classe, vous pouvez utiliser le ifelse() comme vu dans le TP sur la regression logistique. 
# Pour un jeu de données avec 3 classes au moins ce sera un peu différent (cf. exo 4)

###### Question 5: Quelles sont les valeurs des attributs et les classes predites des 5 premiers exemples de l'ensemble de test ?
# Essayez de localiser des 5 premiers exemples de l'ensemble de test sur le graphe représentant la frontiere de décision 
# et les points de l'ensemble de test.
# La classe predite et la probabilité associée vous paraissent-elles coherentes par rapport à ce que vous observez visuellement ?


plot(prediction_test[which(ex3$Classe==0),1:2], col="red", xlim = c(-1.2,1.2), ylim = c(-1.2,1.2))
points(prediction_test[which(ex3$Classe==1),1:2], col="blue")

# Calcul de la performance de ce modèle (nb d'erreurs) sur le jeu de test.
# La fonction evaluate permet de calculer l accuracy (taux de bonnes classifications) du modele sur un jeu de données:
model%>%evaluate(test_x,test_y) # vous devez juste indiquer le jeu de données à évaluer ainsi que les 
# labels associés. Vous obtiendrez le loss et l'accuracy.

###### Question 6: Quelle est l'accuracy de ce modèle sur le jeu de test?



# C'est cette performance (accuracy) qui est une estimation de l'erreur de généralisation de ce modèle.
# En pratique, vous devrez toujours procéder comme suit:
# - séparation apprentissage/test
# - choix d'une structure (nb couche cachées, nb neurones, activations...)
# - apprentissage du modèle en s'arretant avant le surapprentissage (utilisation d'un ensemble de validation)
# - évaluation du modèle sur le test
# Ensuite, vous pouvez changer de structure, et recommencer. Vous pourrez sélectionner ainsi la 
# structure de réseau la plus adpatée à vos données.





##########################################################################
# Exo 4 : Sélection du meilleur réseau sur un jeu de données multiclasse
#########################################################################

dataset = read.table("./data/segment.dat")

# 2310 individus, 19 attributs (colonnes 1 à 19), la classe est dans la colonne 20 (cf description des données dans segment.txt)
# Lorsqu'il y a plus de 2 classes, quelques commandes sont différentes.

# Séparation Apprentissage/Test, classique
nall = nrow(dataset) #total number of rows in data
nall

ntrain = floor(0.80 * nall) # number of rows for train: 80%
ntrain
ntest = floor(0.20* nall) # number of rows for test: 20%
ntest
index = sample(nall) # permutation aléatoire des nombres 1, 2, 3 , ... nall
index

train_x = dataset[index[1:ntrain],1:19] # ensemble d'apprentisssage
train_labels = dataset[index[1:ntrain],20] # labels d'apprentissage

test_x = dataset[index[(ntrain+1):nall],1:19] # ensemble de test
test_labels = dataset[index[(ntrain+1):nall],20] # labels de test

train_x = matrix(unlist(train_x), ncol = 19)
test_x = matrix(unlist(test_x), ncol = 19)

table(train_labels)
# Vous observez que la classe est un entier entre 1 et 7.
# Pour keras, il faut que les labels des classes commencent à zéro
# On va donc soustraire 1 à train_labels et test_labels, et creer train_y et test_y:
train_y = train_labels-1
test_y = test_labels-1
# Puis il faut transformer ces vecteurs pour qu'ils soient au bon format attendu par keras : 
# pour chaque individu, il faut un vecteur de taille 7 (nb de classes) où tous les
# composants sont à 0 sauf celui qui correspond à la classe de l'individu.
# Par exemple, pour un individu de classe 3, ce vecteur doit etre 0 0 0 1 0 0 0 (un 1 pour la classe 3, et 0 pour les autres)
# Ceci est fait par la commande : 
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)



### Créer un réseau de neurones pour ces données, puis lancer le fit
# Attention : 
# 1) les individus sont représentés par 19 attributs donc il doit y avoir 19 neurones sur la couche d'entrée
# 2) Il y a 7 classes possibles donc il doit y avoir 7 neurones sur la couche de sortie
# 3) Pour obtenir des probabilités en sortie, la fonction d'activation de la couche de sortie doit etre 'softmax' (une généralisation de la fonction sigmoide a plusieurs classes)
# 4) Quand il y a plus de 2 classes, il faut mettre loss = 'categorical_crossentropy' dans la commande compile
# au lieu de binary_crossentropy
# 5) N'oubliez pas de mettre l'option view_metrics = F dans la fonction 'fit' pour accélerer l'apprentissage
# 6) Pour visualiser les courbes a la fin de l'apprentissage, on peut stocker les donnees dans la variable 'history':
# history <- model %>% fit(..., view_metrics=F)
# puis afficher sous forme de courbe les informations stockees dans history : 
# plot_NN_loss(history)
# plot_NN_accuracy(history)
# (fonctions definies dans fonctions_utiles.R)

model <- keras_model_sequential()
model %>%
  layer_dense(units = 80, input_shape =19, activation = 'relu') %>%
  layer_dense(units = 40, activation = 'relu') %>%
  layer_dense(units = 30, activation = 'relu') %>%
  layer_dense(units = 7,activation = 'softmax') 

model %>% compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = c('accuracy'))

history <-model %>% fit(train_x, train_y, epochs = 500, batch.size = 10, validation_split = 0.2,callbacks = list(callback_early_stopping("val_loss", patience = 100)),view_metrics = F)

plot_NN_loss(history)
plot_NN_accuracy(history)

###### Question 1: Représentez graphiquement votre réseau.

###### Question 2: Combien a-t-il fallu d'epochs pour apprendre le modèle ? 

### Une fois le modèle appris (dans la variable model par exemple), appelez la commande 'predict' pour obtenir la sortie du reseau pour les exemples de l'ensemble de test.
###### Question 3: quelle est la forme de la prediction pour un exemple ? Quelle est la classe associée a chaque exemple ?
predict(model, test_x)
### La fonction 'evaluate' permet de calculer l'accuracy comme tout à l'heure. Calculez l'accuracy sur l'ensemble de test.
###### Question 4: quelle est l'erreur empirique du modèle ? quelle est l'erreur reelle du modèle ?

model%>%evaluate(test_x,test_y)

model%>%evaluate(train_x,train_y)

### Essayez différentes structures (modèles) et calculer leurs erreurs de généralisation.
