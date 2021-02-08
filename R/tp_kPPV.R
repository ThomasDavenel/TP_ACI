# ---------- Chargement des packages nécéssaires
library(class) # à installer peut etre d'abord
source("./fonctions_utiles.R")

# ---------- Chargement des données
ex1 = read.table("./data/data_exam.txt", header = T)
ex1$Reussite = as.factor(ex1$Reussite)

# ---------- Separation Apprentissage/Validation/Test
nall = nrow(ex1) #total number of rows in data
ntrain = floor(0.7 * nall) # number of rows for train: 70% (vous pouvez changer en fonction des besoins)
nvalid = floor(0.15 * nall) # number of rows for valid: 15% (idem)
ntest = nall - ntrain - nvalid # number of rows for test: le reste

set.seed(20) # choix d une graine pour le tirage aléatoire
index = sample(nall) # permutation aléatoire des nombres 1, 2, 3 , ... nall

train = ex1[index[1:ntrain],] # création du jeu d'apprentissage
valid = ex1[index[(ntrain+1):(ntrain+nvalid)],] # création du jeu de validation
test = ex1[index[(ntrain+nvalid+1):(ntrain+nvalid+ntest)],] # création du jeu de test

# ----- Visualisation des données apprentissage et validation

#Affichage des données d apprentissage
plot(train[which(train$Reussite==0),1:2], col = "red", xlim = c(0,105), ylim = c(0,105))
points(train[which(train$Reussite==1),1:2], col = "blue")

# Affichage des données de validation (on les affiche avec un triangle , pch = 2)
points(valid[which(valid$Reussite==1),1:2], col = "blue", pch = 2)
points(valid[which(valid$Reussite==0),1:2], col = "red", pch = 2)



##################### Exercice 1  #############################
#                Utilisation du KNN                           #
###############################################################
# On commence par k=1
# Avec le knn pas de contruction de modèle, on peut faire directement les prévisions à partir de l'ensemble d'apprentissage
# La commande est la fonction knn avec comme paramètres : 
# - l'ensemble d'apprentissage (seulement les données, pas les classes)
# - l'ensemble que l'on veut prédire (seulement les données aussi)
# - un vecteur contenant les classes de l'ensemble d'apprentissage
# - le k du k-ppv (nombre de ppv pour prédire)

pred = knn(train[,1:2], valid[,1:2],train$Reussite,k=1)


### TODO: Calculez l'erreur sur l'ensemble de validation ?
# Aide : sum( ... )





# ---------- Visualisation de la frontière de décision
# Le troisième paramètre représente le k du k-PPV (vous le changerez si besoin)
dessiner_frontiere_knn(train, valid,k=1,0,105,0,105,c("red", "blue"))



##################### Exercice 2  #############################
#           Selection de l'hyperparamètre k                   #
#           avec l'ensemble de validation                     #
###############################################################
# Comme pour les SVMs, on utilise l'ensemble de validation pour fixer les hyperparametres de la methode.

### TODO: Recommencer en essayant plusieurs valeurs de k (2,3,5,10,15,20 par exemple) et choisir la valeur la plus adaptée,
# i.e. celle pour laquelle l'estimation de l'erreur reelle sur l'ensemble de validation est la plus petite.






### TODO: Le parametre k ayant ete fixe a la question precedente : 
# - visualisez la frontiere de decision obtenue, en affichant les donnees d'apprentissage et de test
# - estimez l'erreur de généralisation faite par le K-NN





##################### Exercice 3  #############################
#          Weigthed distance knn                              #
###############################################################
install.packages("kknn")
install.packages("caret")
library(kknn) #Weighted k-Nearest Neighbor Classifier 
library(caret) #Misc functions for training and plotting classification and regression models

# ---------- Chargement des données
dataset<- read.table("./data/wine.data", sep = ",")
head(dataset)
dataset$V1=as.factor(dataset$V1)
names(dataset)[1]<-"class" # renommage de la variable V1 (classe)

# ---------- Separation Apprentissage/Test (avec caret)
train_index<-createDataPartition(dataset$class,p=0.7,list=F)
train <- dataset[train_index,]
test <-dataset[-train_index,]


# La fonction kknn implémente le Weighted k-Nearest Neighbor Classifier
# Les paramètres sont : 
# - Distance = le parametre de la distance de Minkowski
# - k = le nombre de voisins
# - kernel = le noyau à utiliser pour pondérer les distances
# "rectangular" correspond au knn non pondéré
# "triangular" a une ponderation lineairement decroissante en fonction de la distance
pred_wknn=kknn(class~., train, test, distance = 2, k=5, kernel = "rectangular")

### Question 1: Quelle est le nom de la distance de Minkowski utilisée ici ?


# ---------- Performances
pred_wknn$fitted.values # vecteur des predictions
# TODO : Calculer le taux de bonnes classification





### Question 2: comparer les taux de bonnes classifications obtenus avec et sans ponderation de la distance.



### Question 3: Observez les données. Que constatez-vous ?
summary(dataset)



# ---------- Pretraitement des données (avec caret)
preproc_params<- preProcess(dataset[,2:14], method=c("range"))
scaled_data<-predict(preproc_params,dataset[,2:14])
scaled_data$class=dataset$class

### Question 4: Quel est le resultat de ce pre-traitement sur les donnees ?
summary(scaled_data)

train_index<-createDataPartition(dataset$class,p=0.7,list=F)
train <- scaled_data[train_index,]
test <-scaled_data[-train_index,]


### Question 5: comparer les taux de bonnes classifications obtenus avec et sans ponderation de la distance.


