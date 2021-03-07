# Installation package pour libSVM
install.packages('e1071')
library(e1071)

# ----- Chargement des fonctions utiles
source("./fonctions_utiles.R")


##################### Exercice 1 ################
#      SVM Linéaire - Influence de C            #
#################################################



# ----- Chargement des données
dataset = read.table("./data/LineaireNoisy.txt", header = T, dec = ".")

# Inspect the data type and levels of the attributes of the dataset
###### Question 1
# Combien d'exemples ? d'attributs ? de classes ? Repartition des exemples par classe ? valeurs des classes ?
head(dataset)
str(dataset)
# L'application de la commande table() au vecteur contenant les labels (colonne y de dataset) permet de déterminer
# le nombre d'individus appartenant aux différentes classes.
table(dataset$y)


# Affichage des données
plot(dataset[which(dataset$y==0),1:2], col="red", xlim = c(0,20), ylim = c(0,20.2))
points(dataset[which(dataset$y==1),1:2], col="blue")

###### Question 2
# En vous aidant d'une règle ou d'une feuille de papier, montrer que l'ensemble de données dans dataset
# est linéairement séparable.
# Identifiez également quelques individus qui sont proches de la frontière linéaire.


# ----- Estimation d'un modele SVM sur les données d apprentissage
svm_hard <- svm(y~.,data = dataset, kernel='linear',cost=1000,type='C-classification', scale = F)
# Vous devez donner comme paramètres de svm : 
# - une formule pour indiquer quelle est la classe et quelles sont les données d'apprentissage 
#   (y~.) signifie que y est la classe cible et que toutes les autres colonnes sont les descripteurs
# - le type de noyau utilisé (pour l instant linear)
# - le paramètre cost (qui correspond au C du cours), on prend 1000 pour l'instant
# - type ='C-classification' pour signifier que l'on fait de la classification (et non de la régression)
# - scale = F signifie qu'on ne fait pas de centrage-réduction sur les données
# pour faciliter la compréhension de ce premier exo

# Affichage des informations du modèle
summary(svm_hard)
# Vous retrouvez ici les différentes infos que l'on a données en créant le modèle, 
# ainsi que le nombre de vecteurs supports

###### Question 3:
# Combien y a t il de vecteurs supports pour ce modèle ?


###### Question 4:
# Quelle est la fonction du paramètre C ? Quelle doit être son influence lorsqu'on 
# lui donne une grande valeur ?



# ----- Analyse du modele
### Affichage de la frontière et des marges
dessiner_marge(dataset, svm_hard, 0,20,0,20.5, c("red", "blue"))
# Paramètres :
# - les données de l'ensemble apprentissage
# - le modèle appris par svm()
# - les bornes du graphique en abscisses puis ordonnées
# - les couleurs associées aux différentes classes

# Remarque : les vecteurs support sont représentés par des croix encerclées

###### Question 5:
# (a) Que pensez-vous de cette frontière et ces marges ?
# (b) Combien de points sont à l'intérieur des marges ?
# (c) Combien de points sont mal classés par ce modèle ?



# On peut récupérer également des informations plus précises sur le modèle qui vont nous permettre 
# de comprendre son processus de décision
print(svm_hard$index) # les indices de ligne des vecteurs support dans dataset
print(svm_hard$SV) # les vecteurs support (VS)
print(svm_hard$coefs) # les coefficients de Lagrange associés à chaque VS
# (alpha_i du cours) * u_i (classe du VS)
print(svm_hard$rho) # the negative intercept, ie la constante du modèle (-w0 du cours)




#  ----- Classification d'un vecteur par le modèle SVM appris

# Rappel : la prise de décision associée à un SVM lineaire est la suivante : 
# pour un individu x de coordonnées (x1, x2), on calcule la fonction de décision
# h(x) = alpha1 * u1 * SV1.x + alpha2 * u2* SV2.x + alpha3 * u3 * SV3.x - w0, où
# les alpha_i sont les coeffcients de Lagrange
# les SVi sont les vecteurs support
# le . représente le produit scalaire
# les u_i sont les classes des VS
# w0 est la constante du modèle.
# Il y a 3 termes dans cet exemple pour h(x) car il y a 3 vecteurs supports
# Ensuite, la décision du SVM (classe) dépend du signe de h(x).
# La frontière de décision est donnée par h(x) = 0 (c'est une droite dans le cas
# du SVM linéaire en dimension 2, on la représentera tout à l'heure)


### TODO: Calculer (en utilisant les commandes R ci dessus), la valeur de h(x) pour le premier 
# individu de dataset (dataset[1,1:2]) (ou n'importe quel autre)
# Aide: pour deux vecteurs u et v, le produit scalaire u.v peut s'écrire sum(u*v)
h=svm_hard$coefs[1,1]*sum(svm_hard$SV[1,1:2]*dataset[1,1:2])+svm_hard$coefs[2,1]*sum(svm_hard$SV[2,1:2]*dataset[1,1:2])+svm_hard$coefs[3,1]*sum(svm_hard$SV[3,1:2]*dataset[1,1:2])-svm_hard$rho
print(h)
###### Question 6:
# Quelle est la décision que vous prenez alors pour cet individu ?



###### Question 7: 
# Calculer maintenant h(x) pour le premier vecteur support du modèle.
# Vous devriez obtenir une valeur "particulière". Est ce normal ?



# Les valeurs de h(x) pour tous les individus de l'ensemble d'apprentissage
# peuvent s'obtenir directement à l'aide de la commande suivante : 
svm_hard$decision.values
### Vérifiez que les valeurs de h(x) que vous avez trouvées au dessus sont correctes.





# ----- Influence du parametre C
# Modifier le paramètre cost lors de l'apprentissage du SVM

### TODO: Apprendre un nouveau modèle avec cost = 1

svm_hard_cost1 <- svm(y~.,data = dataset, kernel='linear',cost=1,type='C-classification', scale = F)
summary(svm_hard_cost1)
dessiner_marge(dataset, svm_hard_cost1, 0,20,0,20.5, c("red", "blue"))
print(svm_hard_cost1$index) # les indices de ligne des vecteurs support dans dataset
print(svm_hard_cost1$SV) # les vecteurs support (VS)
print(svm_hard_cost1$coefs) # les coefficients de Lagrange associés à chaque VS
# (alpha_i du cours) * u_i (classe du VS)
print(svm_hard_cost1$rho) # the negative intercept, ie la constante du modèle (-w0 du cours)
###### Question 8:
# (a) Combien de vecteurs supports contient ce nouveau modele ?
# (b) Dessiner la nouvelle frontière et ses marges.
# (c) Combien de points sont à l'intérieur des marges ?
# (d) Combien de points sont mal classés par ce modèle ?  
# (e) Quelles sont les valeurs des coefficients de Lagrange (alpha) pour ces points (dans la marge ou mal classés) ?
# (f) Est ce normal ?


### TODO: Mêmes questions pour un nouveau modele appris avec cost = 0.01


svm_hard_cost2 <- svm(y~.,data = dataset, kernel='linear',cost=0.01,type='C-classification', scale = F)
summary(svm_hard_cost2)
dessiner_marge(dataset, svm_hard_cost2, 0,20,0,20.5, c("red", "blue"))
print(svm_hard_cost2$index) # les indices de ligne des vecteurs support dans dataset
print(svm_hard_cost2$SV) # les vecteurs support (VS)
print(svm_hard_cost2$coefs) # les coefficients de Lagrange associés à chaque VS
# (alpha_i du cours) * u_i (classe du VS)
print(svm_hard_cost2$rho) # the negative intercept, ie la constante du modèle (-w0 du cours)


##################### Exercice 2#################################
#      Comparaison SVM / Régression logistique                  #
#################################################################

# ----- Chargement des données
exo2 = read.table("./data/exo2_SVM.txt", header = T)

# Individus décrits par 2 attributs quantitatifs x1 et x2
# y est la classe : 0 ou 1


#  ----- Apprentissage d'un modèle SVM 
svm_hard_cost <- svm(y~.,data = exo2, kernel='linear',cost=1000,type='C-classification', scale = F)
summary(svm_hard_cost)
dessiner_marge(exo2, svm_hard_cost, 0,20,0,20.5, c("red", "blue"))
print(svm_hard_cost$index) # les indices de ligne des vecteurs support dans dataset
print(svm_hard_cost$SV) # les vecteurs support (VS)
print(svm_hard_cost$coefs) # les coefficients de Lagrange associés à chaque VS
# (alpha_i du cours) * u_i (classe du VS)
print(svm_hard_cost$rho) # the negative intercept, ie la constante du modèle (-w0 du cours)
# Apprendre un modele SVM lineaire avec C = 1000 sur ces données
# 
# Afficher la frontière et les marges et garder la figure ouverte



#  ----- Apprentissage d'un modèle de regression logistique 
# On va maintenant apprendre un modèle de regression logistique sur ces memes données
reglog = glm(y~., data = exo2, family = binomial(link = 'logit'))

# On récupère les coefficients de cette régression logistique:
co = reglog$coefficients
print(co)
# L'équation de la droite de régression logistique s'écrit : 
# co[1] + co[2]*x1 + co[3]*x2 = 0
# ou encore x2 = -co[1]/co[3] - co[2]/co[3] * x1
# La commande abline(a,b) permet de tracer une droite d'ordonnée à l'origine a et de coeff directeur b
# On peut donc ajouter au graphe précédent la droite de regression logistique :
abline(-reglog$coefficients[1]/reglog$coefficients[3],-reglog$coefficients[2]/reglog$coefficients[3], col = "green")


###### Question 1: Conclusion ?




##################### Exercice 3 #################################
#        Choix de C par séparation A/V/T                         #
##################################################################

# ----- Chargement et analyse des données
load("./data/spam7.Rdata")
# Les données se trouvent dans le tableau spam7
# Afficher les 5 premières lignes de ce tableau à l aide de la commande head
# La description de ces donéees se trouve ici : 
# http://math.furman.edu/~dcs/courses/math47/R/library/DAAG/html/spam7.html

###### Question 1: (répondre en utilisant des commandes R)
# (a) Combien y a t il d individus dans ces données ?
print(nrow(spam7))
# (b) Combien de variables décrivent les individus ?
print(ncol(spam7)-1)

# (c) Quelle est la variable cible ? Combien de classes différentes y a t'il ?
spam7[,7]
# (d) Combien y a t'il d'individus par classe ?
print(nrow(spam7[which(spam7[,7]=='y'),1:5]))
print(nrow(spam7[which(spam7[,7]=='n'),1:5]))



# Pour sélectionner la meilleure valeur de C à utiliser avec ces données, on va 
# séparer le jeu de données en ensembles de : apprentissage/validation/test
# Pour éviter toute mauvaise surprise (par exemple si les individus sont rangés par classe dans le tableau
# de données), il est fortement conseillé de choisir aléatoirement les individus qui composeront
# les trois jeux de données.

### Separation apprentissage / validation / test
nall = nrow(spam7) #total number of rows in data
# Une séparation correcte en ensembles d'apprentissage et de test se fait par sélection aléatoire,
# notamment au cas où les données seraient triées par valeur de classe, afin d'éviter d'avoir tous les exemples d'une même classe 
# dans l'ensemble d'apprentissage et les exemples de l'autre classe dans l'ensemble de test.

ntrain = floor(0.7 * nall) # number of rows for train: 70% (vous pouvez changer en fonction des besoins)
nvalid = floor(0.15 * nall) # number of rows for valid: 15% (idem)
ntest = nall - ntrain - nvalid # number of rows for test: le reste

set.seed(20) # choix d une graine pour le tirage aléatoire
index = sample(nall) # permutation aléatoire des nombres 1, 2, 3 , ... nall

train = spam7[index[1:ntrain],] # création du jeu d'apprentissage
valid = spam7[index[(ntrain+1):(ntrain+nvalid)],] # création du jeu de validation
test = spam7[index[(ntrain+nvalid+1):(ntrain+nvalid+ntest)],] # création du jeu de test


###### Question 2:
# Quelle est la taille de l'ensemble d'apprentissage, de validation et de test ?




# ----- Apprentissage d'un modèle SVM
### TODO: Apprendre un premier modèle avec C = 1 (toujours avec un noyau lineaire)
# Attention, ici il faut mettre scale = T dans les paramètres de la fonction svm, et non scale = F
# Cela centre et réduit les variables.

###### Question 3: Combien a t'il de vecteurs supports ?

mod <- svm(yesno~.,data = train, kernel='linear',cost=1,type='C-classification', scale = T)
summary(svm_hard_cost)

# -----  Calcul de la performance sur l'ensemble d apprentissage
# La commande suivante permet d'obtenir les prédictions faites par le modèle sur les données d'apprentissage
p = predict(mod, train)
res =0
resY =0
mauvaisePY=0
for(i in 1:nrow(train)){
  if(train[i,7]==p[i]){
    res=res+1
    if(train[i,7]=="y"){
      resY=resY+1
    }
  }
  else{
    if(train[i,7]=="y"){
      mauvaisePY=mauvaisePY+1
    }
  }
}
print("nombre de bonnes predictions")
print(res)

print("nombre de bonnes predictions sur Y")
print(resY)

print("nombre de bonnes predictions sur N")
print(res-resY)

print("nombre de mauvaise predictions")
print(nrow(train)-res)

print("nombre de mauvaise predictions sur Y")
print(mauvaisePY)

print("nombre de mauvaise predictions sur N")
print(nrow(train)-res -mauvaisePY)

### TODO:
# Calculer le nombre de prédictions correctes faites par ce modèle sur l'ensemble d'apprentissage
# Aide : comme d'habitude (cf tp_Tree), sum(p == ...)

###### Question 4:
# (a) Quel est le pourcentage d'erreur sur l'ensemble d'apprentissage ?
# (b) Donner la matrice de confusion
# (c) Sur quelle classe ce SVM se trompe t'il plus souvent ?


# ----- Calcul de la perf sur ensemble de validation
### TODO, idem mais avec l'ensemble de validation
###### Question 5: idem mais avec l'ensemble de validation

p = predict(mod, valid)
res =0
resY =0
mauvaisePY=0
for(i in 1:nrow(valid)){
  if(valid[i,7]==p[i]){
    res=res+1
    if(valid[i,7]=="y"){
      resY=resY+1
    }
  }
  else{
    if(valid[i,7]=="y"){
      mauvaisePY=mauvaisePY+1
    }
  }
}
print("nombre de bonnes predictions")
print(res)

print("nombre de bonnes predictions sur Y")
print(resY)

print("nombre de bonnes predictions sur N")
print(res-resY)

print("nombre de mauvaise predictions")
print(nrow(valid)-res)

print("nombre de mauvaise predictions sur Y")
print(mauvaisePY)

print("nombre de mauvaise predictions sur N")
print(nrow(valid)-res -mauvaisePY)



# ----- Choix de C
### TODO: Essayer maintenant différentes valeurs de C (entre 0.001 et 100 par exemple), et 
# calculer pour chacune de ces valeurs le pourcentage d'erreur en apprentissage 
# et en validation
# Aide : Vous pouvez définir un vecteur C (avec la commande c() ) contenant les différentes valeurs que vous voulez tester
# et faire une boucle sur les valeurs de C
# for(i in 1:length(C)){
#   récuperer C[i], créer le modèle, calculer les erreurs
#    ...
#  }
C = c(0.001,0.01,0.1,1,10,100)
for(i in 1:length(C)){
  mod <- svm(yesno~.,data = train, kernel='linear',cost=C[i],type='C-classification', scale = T)
  print("c = ")
  print( C[i])
  p = predict(mod, train)
  res=0
  for(i in 1:nrow(train)){
    if(train[i,7]!=p[i]){
      res=res+1
    }
  }
  print("erreur apprentissage: ")
  print(res/nrow(train))
  p = predict(mod, valid)
  res=0
  for(i in 1:nrow(valid)){
    if(valid[i,7]!=p[i]){
      res=res+1
    }
  }
  print("erreur validation: ")
  print(res/nrow(valid))
  print("~~~~~~~~~~~~")
}



###### Question 6:
# (a) Quelle est la meilleure valeur de C ?
# (b) Donner l'estimation de l'erreur de généralisation faite par le SVM obtenu





##################### Exercice 4  #############################
#           Données pas linéairement séparables               #
###############################################################

# ----- Chargement et analyse des données
dataset = read.table("./data/SepNonLineaire.txt", header = T)
# La variable cible est y

###### Question 1: Inspecter les données (nb de variables, variable cible, nb d individus)

nall = nrow(dataset) #total number of rows in data
ntrain = floor(0.7 * nall) # number of rows for train: 70% (vous pouvez changer en fonction des besoins)
nvalid = floor(0.15 * nall) # number of rows for valid: 15% (idem)
ntest = nall - ntrain - nvalid # number of rows for test: le reste

set.seed(20) # choix d une graine pour le tirage aléatoire
index = sample(nall) # permutation aléatoire des nombres 1, 2, 3 , ... nall

train = dataset[index[1:ntrain],] # création du jeu d'apprentissage
valid = dataset[index[(ntrain+1):(ntrain+nvalid)],] # création du jeu de validation
test = dataset[index[(ntrain+nvalid+1):(ntrain+nvalid+ntest)],] # création du jeu de test

###### Question 2: 
# (a) Quelles sont les tailles des ensembles ?
# (b) Quelle est la repartition des classes dans les differents ensembles ?


# Affichage des données d'apprentissage et de validation (triangles)
plot(train[which(train$y==0),1:2], col="red", xlim = c(-3,4), ylim = c(-5,9))
points(train[which(train$y==1),1:2], col="blue")
points(valid[which(valid$y==0),1:2], col="red", pch = 2)
points(valid[which(valid$y==1),1:2], col="blue", pch = 2)

###### Question 3: que pensez-vous de ces données ?


# ----- Apprentissage d'un SVM linéaire (pour essayer quand meme)
# On essaye C = 1000 pour débuter
### TODO: Apprendre un modèle lineaire et calculer erreurs apprentissage/validation
svm_model <- svm(y~.,data = train, kernel='linear',cost=1000,type='C-classification', scale = F)
summary(svm_model)
p = predict(svm_model, train)
res=0
for(i in 1:nrow(train)){
  if(train[i,3]!=p[i]){
    res=res+1
  }
}
print("erreur apprentissage: ")
print(res/nrow(train))
p = predict(svm_model, valid)
res=0
for(i in 1:nrow(valid)){
  if(valid[i,3]!=p[i]){
    res=res+1
  }
}
print("erreur validation: ")
print(res/nrow(valid))

# Affichage de la frontière avec la fonction dessiner_frontiere_svm
dessiner_frontiere_svm(train[,1:2], train$y, valid[,1:2], valid$y, svm_model, -3,4,-5,9, c("red", "blue"))
# Paramètres : 
# - les données de l'ensemble apprentissage
# - les classes de l'ensemble apprentissage
# - les données de l'ensemble de validation
# - les classes de l'ensemble de validation
# - le modèle appris par svm()
# - les bornes du graphique en abscisses puis ordonnées
# - les couleurs associées aux différentes classes


### TODO: Essayer quelques autres valeurs de C et noter les erreurs d apprentissage et validation
C = c(0.001,0.01,0.1,1,10,100)
for(i in 1:length(C)){
  mod <- svm(y~.,data = train, kernel='linear',cost=C[i],type='C-classification', scale = F)
  print("c = ")
  print( C[i])
  p = predict(mod, train)
  res=0
  for(i in 1:nrow(train)){
    if(train[i,3]!=p[i]){
      res=res+1
    }
  }
  print("erreur apprentissage: ")
  print(res/nrow(train))
  p = predict(mod, valid)
  res=0
  for(i in 1:nrow(valid)){
    if(valid[i,3]!=p[i]){
      res=res+1
    }
  }
  print("erreur validation: ")
  print(res/nrow(valid))
  print("~~~~~~~~~~~~")
}


# ----- Apprentissage d'un SVM à Noyau
#  On va maintenant utiliser un noyau gaussien (avec gamma = 1 pour commencer, et cost = 1)
svm_model <- svm(y~.,data = train, kernel='radial',cost=1,type='C-classification', gamma = 1)

### TODO: dessiner la frontière associée à ce SVM
dessiner_frontiere_svm(train[,1:2], train$y, valid[,1:2], valid$y, svm_model, -3,4,-5,9, c("red", "blue"))

###### Question 4: Calculer le nombre d erreurs apprentissage et validation


### TODO: Faire varier le gammma entre 0.01 et 2 et calculer les erreurs 
gamma = c(0.01,0.1,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2)
C = c(0.001,0.01,0.1,1,10,100)
for(idxC in 1:length(C)){
  for(idxGamma in 1:length(gamma)){
    mod <- svm(y~.,data = train, kernel='radial',cost=C[idxC],type='C-classification', gamma = gamma[idxGamma])
    print("C = ")
    print( C[idxC])
    print("gamma = ")
    print( gamma[idxGamma])
    p = predict(mod, train)
    res=0
    for(i in 1:nrow(train)){
      if(train[i,3]!=p[i]){
        res=res+1
      }
    }
    print("erreur apprentissage: ")
    print(res/nrow(train))
    p = predict(mod, valid)
    res=0
    for(i in 1:nrow(valid)){
      if(valid[i,3]!=p[i]){
        res=res+1
      }
    }
    print("erreur validation: ")
    print(res/nrow(valid))
    print("~~~~~~~~~~~~")
  }
}
### TODO: Puis faire varier cost ET gamma (double boucle)

###### Question 5: 
# (a)Quelle est le meilleur couple (cost, gamma) pour ces données ?
# (b) Quelle estimation de l'erreur de généralisation pouvez vous faire ?




# On va essayer maintenant un noyau polynomial (de degré 2 pour commencer)
svm_model <- svm(y~.,data = train, kernel='polynomial',cost=1,type='C-classification', degree = 2, coef0 = 1)

### TODO: Reprenez les mêmes questions qu'avec le noyau gaussien en faisant varier le degré (entre 2 et 5) ET le cost

gamma = c(2,2.5,3,3.5,4,4.5,5)
C = c(0.1,1,10,100)
for(idxC in 1:length(C)){
  for(idxGamma in 1:length(gamma)){
    mod <- svm(y~.,data = train, kernel='radial',cost=C[idxC],type='C-classification', gamma = gamma[idxGamma])
    print("C = ")
    print( C[idxC])
    print("gamma = ")
    print( gamma[idxGamma])
    p = predict(mod, train)
    res=0
    for(i in 1:nrow(train)){
      if(train[i,3]!=p[i]){
        res=res+1
      }
    }
    print("erreur apprentissage: ")
    print(res/nrow(train))
    p = predict(mod, valid)
    res=0
    for(i in 1:nrow(valid)){
      if(valid[i,3]!=p[i]){
        res=res+1
      }
    }
    print("erreur validation: ")
    print(res/nrow(valid))
    print("~~~~~~~~~~~~")
  }
}

