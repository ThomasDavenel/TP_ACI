# ----- Chargement des fonctions utiles
install.packages("gtools")
library(gtools)
source("fonctions_utiles.R")
#install.packages("e1071")
#library("e1071")

#-----------------------------------
# Exercice 1: Regression logistique "lineaire"
#-----------------------------------


# ----- Chargement des données
ex1 = read.table("./data/data_exam.txt", header = T)

# Inspect the data type and levels of the attributes of the dataset
###### Question 1: Combien d'exemples ? d'attributs ? de classes ? Repartition des exemples par classe ?
head(ex1)
str(ex1)
table(ex1$Reussite)


# ----- Separation Apprentissage/Test 
# Une séparation correcte en ensembles d'apprentissage et de test se fait par sélection aléatoire,
# notamment au cas où les données seraient triées par valeur de classe, afin d'éviter d'avoir tous les exemples d'une même classe 
# dans l'ensemble d'apprentissage et les exemples de l'autre classe dans l'ensemble de test.
nall = nrow(ex1) #total number of rows in data
ntrain = floor(0.7 * nall) # number of examples (rows) for train: 70% (vous pouvez changer en fonction des besoins)
ntest = nall - ntrain # number of examples for test: le reste

set.seed(20) # choix d'une graine pour le tirage aléatoire
index = sample(nall) # permutation aléatoire des nombres 1, 2, 3 , ... nall

ex1_app = ex1[index[1:ntrain],] # création du jeu d'apprentissage
ex1_test = ex1[index[(ntrain+1):(ntrain+ntest)],] # création du jeu de test

###### Question 2: quel est le nombre d'exemples dans les ensembles d'apprentissage et de test ?
print(nrow(ex1_app))
print(nrow(ex1_test))

# ----- Affichage du jeu de données 

#Affichage des données d apprentissage
# D'abord ceux qui ont échoués (pour lesquels ex1$Reussite = 0) en rouge
plot(ex1_app[which(ex1_app$Reussite==0),1:2], col = "red", xlim = c(0,105), ylim = c(0,105))
# puis ceux qui ont réussis, en bleu
points(ex1_app[which(ex1_app$Reussite==1),1:2], col = "blue")

# Affichage des données de test (on les affiche avec un triangle , pch = 2)
points(ex1_test[which(ex1_test$Reussite==1),1:2], col = "blue", pch = 2)
points(ex1_test[which(ex1_test$Reussite==0),1:2], col = "red", pch = 2)


# ----- Estimation du modele de regression logistique sur les données d apprentissage
# On utilise la fonction glm. On doit lui indiquer le nom de la colonne dans laquelle est stockée la variable
# cible, ici Reussite. On utilise '.' pour dire qu'on utilise toutes les autres colonnes pour construire le classifieur.
# Le terme "family = binomial(link = 'logit')" sert à préciser que l'on fait de la régression logistique
reg_ex1 = glm(Reussite~., data = ex1_app, family = binomial(link = 'logit'))

# Affichage du modèle

###### Question 3: 
# Quels sont les coefficients estimés ? Quelle est l'équation du modèle associé ? (slides 14-15 du cours)




### TODO: Calculez à la main (enfin en utilisant la console R quand même) la sortie du modèle associée au premier
# étudiant de l'ensemble d'apprentissage.
# Aide : il suffit d'appliquer l'équation du modèle avec les données de l'étudiant. Regardez les fonctions sigmoid et inv.logit.
x = reg_ex1$coefficients[1] + ex1_app[1,1]*reg_ex1$coefficients[2]+ex1_app[1,2]*reg_ex1$coefficients[3]
print(sigmoid(x))


###### Question 4: Cette valeur correspond à la probabilité que cet étudiant soit de la classe 1 (Réussite=1)
# Quelle décision allez-vous prendre pour cet étudiant? 



# ----- Calcul de l'erreur empirique (sur donnees d apprentissage) de ce modele

idx = 1 # indice de l'élément qui nous interesse (vous pourrez en essayer d'autres)
# On peut prédire les classes des étudiants automatiquement avec la fonction predict. 
# Cette fonction réalise exactement le calcul que vous venez de faire à la main.
# La valeur renvoyée par cette commande  correspond à la probabilité que l étudiant en question soit de la classe 1 (donc qu'il ait réussi l examen final). 
# C'est donc 1-P(Reussite=0)
p = predict(reg_ex1, ex1_app[idx,], type = 'response')


### TODO: Ecrivez une commande qui permet de déterminer la classe à partir de la probabilité obtenue ci-dessus. 
# Aide : commande ifelse(condition, reponse si oui, reponse si non)
print("etudiant :" )
print(idx)
if(p>0.5){
  print(1)
}else{
  print(0)
}
### TODO: Estimez la classe de chacun des étudiants du jeu d'apprentissage. 
# Aide : appliquer la commande predict à tout le jeu d'apprentissage, puis ifelse de la même façon
for(idx in 1:nrow(ex1_app)){
  p = predict(reg_ex1, ex1_app[idx,], type = 'response')
  print("etudiant :" )
  print(idx)
  if(p>0.5){
    print(1)
  }else{
    print(0)
  }
}

###### Question 5: Combien d'erreurs de prediction sont faites par ce modele sur le jeu d'apprentissage ? Calculez l'erreur empirique (en %).
# Les commandes suivantes vous seront surement utiles : 
# - la commande != permet de comparer les éléments deux à deux de deux vecteurs et dire s'ils sont différents
# - sum() appliquée à un vecteur booléen permet de compter le nombre d'éléments égaux à TRUE dans ce vecteur
# Ici vous devez comparer le vecteur contenant vos prédictions avec la vérité (colonne Reussite de ex1_app) et compter combien de prédictions sont fausses

# A REMPLIR
res=0
for(idx in 1:nrow(ex1_app)){
  p = predict(reg_ex1, ex1_app[idx,], type = 'response')
  if(p>0.5){
    p=1
  }else{
   p=0
  }
  if(p!=ex1_app[idx,3])
    res=res+1
}
print(res)
print((res/nrow(ex1_app))*100)
###### Question 6: Identifiez un étudiant pour lequel le modèle se trompe.
# Aide: vous pouvez utiliser la commande which()
for(idx in 1:nrow(ex1_app)){
  p = predict(reg_ex1, ex1_app[idx,], type = 'response')
  if(p>0.5){
    p=1
  }else{
    p=0
  }
  if(p!=ex1_app[idx,3])
    print(idx)
}

###### Question 7: Donner la matrice de confusion sur le jeu d'apprentissage. 
# Aide : commande table avec 2 paramètres : les prédictions et la vérité. En ligne, les classes prédites et en colonne, les vraies classes.
idx=59
print( predict(reg_ex1, ex1_app[idx,], type = 'response'))
print(ex1_app[idx,3])

res0=0
res1=0
for(idx in 1:nrow(ex1_app)){
  p = predict(reg_ex1, ex1_app[idx,], type = 'response')
  if(p>0.5){
    p=1
  }else{
    p=0
  }
  if(p==ex1_app[idx,3] && p==1)
    res1=res1+1
  if(p==ex1_app[idx,3] && p==0)
    res0=res0+1
}
print(res0)
print(res1)
###### Question 8: Combien d'étudiants (du jeu d'apprentissage) ont réussi l'examen final alors que le modèle prévoyait l'inverse?

# A REMPLIR


###### Question 9: Estimez la classe de chacun des étudiants du jeu de test et calculez l'erreur
# faite par ce modele sur le jeu de test. Aide : idem 
reg_ex1 = glm(Reussite~., data = ex1_test, family = binomial(link = 'logit'))
res=0
for(idx in 1:nrow(ex1_test)){
  p = predict(reg_ex1, ex1_test[idx,], type = 'response')
  if(p>0.5){
    p=1
  }else{
    p=0
  }
  if(p!=ex1_app[idx,3])
    res=res+1
}
print(res)
print(nrow(ex1_test))
print((res/nrow(ex1_test))*100)


# Tracé de la frontière de décision
# La fonction remplissage (donnée dans le fichier fonctions_utiles, allez voir les explications) vous permet de visualiser la frontière de décision.
# Le principe : prédire tous les points du plan et les afficher avec la couleur associée à la prédiction.
# Vous remarquerez que les données d'apprentissage et de test sont également affichées
# Ici le paramètre puissance est mis à 1, les couleurs sont rouge et bleus (vous pouvez changer), et le plan est entre 0 et 105 pour abscisses et ordonnées
dessiner_frontiere_reglog(ex1_app,ex1_test, reg_ex1,1,"red","blue",0,105,0,105)

###### Question 10:
# De quel type est la frontière associéé à la régression logistique ?

# Repérez (visuellement) les étudiants de l'ensemble d'apprentissage qui sont mal classés par le modèle
# Même question avec les étudiants de l'ensemble de test





#-----------------------------------
# Exercice 2: Regression logistique "non lineaire"
#-----------------------------------

# Pour obtenir une sufrace séparatrice non linéaire, on peut ajouter des attributs, en general
# obtenus par application de fonctions aux attribut existants.
# Cela revient à modifier l'espace de description (et à augmenter sa dimension).


# ----- Création de nouvelles variables (élévation à la puissance p des variables initiales)
# ----- p = 2 

# Au départ, nous avons Note1 et Note2, on va créer 3 nouvelles variables : Note1^2, Note2^2 et Note1*Note2
# Ceci est fait par la fonction polynomial (donnée dans fonctions_utiles.R), qui prend un jeu de données en 
# paramètres et une valeur p, et éleve les variables de départ à la puissance p (et combinaisons)

# Création des nouveaux jeux de données (apprentissage et test)
ex1_d2_app = polynomial(ex1_app,2)
ex1_d2_test = polynomial(ex1_test,2)

# Affichage du début de ex1_d2_app
head(ex1_d2_app)

###### Question 1: 
# Comparer (manuellement) la première ligne de ex1_app et la première ligne de ex1_d2_app.
# Quel est le nombre d'attributs ? (dimension du vecteur)
# Dans quelles colonnes sont mises les nouvelles variables ? 
# remarque: la variable cible (Reussite) est toujours en dernière colonne. Repérez l'indice de cette dernière colonne.

# Calcul du modèle de regression logistique associé aux données d apprentissage
reg_ex1_d2 = glm(Reussite~., data = ex1_d2_app, family = binomial(link = 'logit'))
print(reg_ex1_d2$coefficients)

# Tracé de la frontière de décision
dessiner_frontiere_reglog(ex1_app,ex1_test,reg_ex1_d2,2,"red","blue",0,105,0,105)

### TODO: Calculer le nb d erreur que fait ce modèle sur les donnes d apprentissage et test.
# (en utilisant la fonction calcul_erreur écrite précédemment. 
# Attention à donner le bon idx_classe en paramètre). Est-il meilleur que le modèle initial ?
print("Apprentissage")
reg_ex1_d2 = glm(Reussite~., data = ex1_d2_app, family = binomial(link = 'logit'))
res=0
for(idx in 1:nrow(ex1_d2_app)){
  p = predict(reg_ex1_d2, ex1_d2_app[idx,], type = 'response')
  if(p>0.5){
    p=1
  }else{
    p=0
  }
  if(p!=ex1_d2_app[idx,6])
    res=res+1
}
print(res)
print(nrow(ex1_d2_app))
print((res/nrow(ex1_d2_app))*100)
print("Teste")
reg_ex1_d2 = glm(Reussite~., data = ex1_d2_test, family = binomial(link = 'logit'))
res=0
for(idx in 1:nrow(ex1_d2_test)){
  p = predict(reg_ex1_d2, ex1_d2_test[idx,], type = 'response')
  if(p>0.5){
    p=1
  }else{
    p=0
  }
  if(p!=ex1_d2_test[idx,6])
    res=res+1
}
print(res)
print(nrow(ex1_d2_test))
print((res/nrow(ex1_d2_test))*100)

#-----------------------------------
# Exercice 3: Regularisation
#-----------------------------------
install.packages("glmnet")
library(glmnet) # GLM with lasso or elasticnet regularization

# les donnees d'entree doivent etre sous forme de matrices
x_app <- model.matrix(Reussite~., ex1_app)[,-1]
x_test <- model.matrix(Reussite ~., ex1_test)[,-1]

# objective function for binomial model :
# -loglik/nobs + λ*penalty
# with penalty (1-α)/2||β_j||_2^2 + α||β_j||_1

# alpha = elasticnet mixing parameter :
# alpha = 1 <=> lasso regression 
# alpha = 0 <=> ridge regression

# Lasso regularization
model_lasso = glmnet(x_app, ex1_app$Reussite, family = "binomial", alpha = 1, lambda=0.1)
# Display regression coefficients
coef(model_lasso)

# Make predictions on the test data
pred_lasso=predict(model_lasso,x_test,type="response") #  fitted probabilities for "binomial" model
pred_lasso=predict(model_lasso,x_test,type="class") # class label corresponding  to  the  maximum  probability


# TODO: Compute the model accuracy



# TODO : change the value of lambda by one order of magnitude and repeat the evaluation



# TODO : Repeat for Ridge regularization, elasticnet regularization and different values of lambda
# Summarize the results on a table



