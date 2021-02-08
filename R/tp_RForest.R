# ---------- Chargement des packages nécéssaires
library(rpart)
library(rpart.plot)
#install.packages("randomForest")
library(randomForest)
source("fonctions_utiles.R")



#-----------------------------------
# Exercice 1: Prise en main sur un petit jeu de donnees
#-----------------------------------


# ---------- Chargement des données
ex1 = read.table("./data/data_exam.txt", header = T)
# Remember that some machine learning methods (trees, bayes, knn...) require the class to be of type 'factor' to work.
# Other don't (SVM, logistic regression, NN).
ex1$Reussite = as.factor(ex1$Reussite) # convert the class in 'factor' type, needed for datasets which class is not a factor


# Inspect the data type and levels of the attributes of the dataset
###### Question 0: Combien d'exemples ? d'attibuts ? de classes ? Repartition des exemples par classe ?



# ---------- Separation Apprentissage/Validation/Test
nall = nrow(ex1) #total number of rows in data
ntrain = floor(0.7 * nall) # number of rows for train: 70% (vous pouvez changer en fonction des besoins)
ntest = nall - ntrain # number of rows for test: le reste

set.seed(20) # choix d une graine pour le tirage aléatoire
index = sample(nall) # permutation aléatoire des nombres 1, 2, 3 , ... nall

ex1_app = ex1[index[1:ntrain],] # création du jeu d'apprentissage
ex1_test = ex1[index[(ntrain+1):(ntrain+ntest)],] # création du jeu de test


# ----- Visualisation des données apprentissage et validation

#Affichage des données d apprentissage
plot(ex1_app[which(ex1_app$Reussite==0),1:2], col = "red", xlim = c(0,105), ylim = c(0,105))
points(ex1_app[which(ex1_app$Reussite==1),1:2], col = "blue")
# Affichage des données de validation (avec un triangle , pch = 2)
points(ex1_test[which(ex1_test$Reussite==1),1:2], col = "blue", pch = 2)
points(ex1_test[which(ex1_test$Reussite==0),1:2], col = "red", pch = 2)



# ---------- Construction de la foret

# La fonction randomForest permet d'ajuster une forêt aléatoire à partir d'un ensemble d'apprentissage.
# Les paramètres sont :
# - une formule  : NomdeColonne~., ou NomDeColonne correspond au nom de la colonne qui contient la classe
# dans les données (ici Reussite), et le  '.' indique que toutes les autres colonnes sont utilisees comme 
# variables pour construire le classifieur.
# - data = ... , le nom de la variable qui contient les données
# - ntree, qui correspond au nombre d'arbre que vous voulez créer dans la foret (ici 2 dans cet exemple)
foret = randomForest(Reussite~., data = ex1_app, ntree = 2, norm.votes=FALSE)


# Malheureusement, on ne peut (facilement) visualiser les arbres de la foret avec ce package. 
# On peut simplement les afficher textuellement dans la console via la commande suivante : 
getTree(foret, idx) # avec idx le numéro de l'arbre que vous voulez regarder

###### Question 1: en utilisant l'aide (help(getTree)), représentez graphiquement le premier arbre de la foret




# ---------- Inspection de la foret
# L'objet retourné contient differentes informations:
print(foret)

###### Question 2: 
# (a) indiquez quel est le nombre d'attributs tirés aléatoirement a chaque noeud pour selectionner le meilleur predicteur.
# (b) qu'est-ce que les donnees OOB ? (cf. cours slides 67-68)
# (c) quelle est la valeur de l'erreur ? S'agit-il de l'erreur empirique ou de l'erreur reelle ?






###TODO: en utilisant l'aide (help(randomForest)),
# (a) afficher les classes prédites pour les echantillons OOB.
# (b) afficher pour chaque echantillon OOB, le nombre de votes pour chaque classe




###### Question 3:
# (a) Combien d'exemples n'ont jamais ete utilises pour construire l'un des arbres de la foret ?
# (b) Combien d'exemples n'ont jamais fait parti de l'ensemble OOB (pour aucun des arbres de la foret) ?
# Aide: utiliser l'aide (help(randomForest)), et les commandes which(...) ou sum(...)





# ----------  Prediction avec une foret

# La fonction predict marche avec une foret
# Vous pouvez prédire la classe de tous les individus de l'ensemble d'apprentissage d'un coup:
predict(foret, ex1_app)

###### Question 4: Calculer la proportion de prédictions correctes faite par cet arbre sur l'ensemble d'apprentissage
# Aide : commande sum(predict(....)==...) comme pour les autres TPs


### TODO: Relancer la prédiction de la classe de tous les individus de l'ensemble d'apprentissage et le calcul de l'erreur.

###### Question 5: Qu'observez-vous ? Comment expliquez-vous ces résultats ?
# Aide : voir l'aide (help(predict.randomForest)) et la commande predict(foret, ex1_app, type="vote")



###### Question 6: Estimez l'erreur en generalisation (erreur reelle).



# ---------- Affichage de la frontière de décision
# Comme pour les arbres
dessiner_frontiere_foret(ex1_app, ex1_test, foret, 0,105,0,105,c("red", "blue"))


# ---------- Choix du nombre d'arbre dans la foret
### TODO: Recommencez avec 3 arbres






###### Question 7:
# (a) Quelle est l'erreur OOB ? l'erreur en generalisation ?
# (b) Combien d'exemples n'ont jamais ete utilises pour construire l'un des arbres de la foret ?
# (c) Combien d'exemples n'ont jamais fait parti de l'ensemble OOB (pour aucun des arbres de la foret) ?
# (d) Conclure






# ---------- Comparaison avec un arbre de decision
### TODO: construire un arbre de decision (elague bien sûr !) et evaluer ses performances sur l'ensemble de test
tr=...
trOpt=...




### TODO: Comparer les frontieres de decision produites avec celles de la foret
dessiner_frontiere_tree(ex1_app, ex1_test, trOpt, 0,105,0,105,c("red", "blue"))

###### Question 8: Comparez les résultats des RF avec un arbre de décision sur le meme jeu de données (performances et surfaces de décision).



#-----------------------------------
# Exercice 2: En pratique
#-----------------------------------

# On va a present utiliser un plus gros jeu de donnees
ex2 = read.table("./data/segment.dat")
ex2$V20 = as.factor(ex2$V20) # convert the class in 'factor' type


# Inspect the data type and levels of the attributes of the dataset
# Les donnees sont decrites dans le fichier 'segment.txt'
###### Question 0:  Combien d'exemples ? d'attributs ? de classes ? Repartition des exemples par classe ?




# ---------- Separation Apprentissage/Validation/Test
nall = nrow(ex2) #total number of rows in data
ntrain = floor(0.7 * nall) # number of rows for train: 70% 
ntest = nall - ntrain # number of rows for test: le reste

set.seed(20) # choix d une graine pour le tirage aléatoire
index = sample(nall) # permutation aléatoire des nombres 1, 2, 3 , ... nall

ex2_app = ex2[index[1:ntrain],] # création du jeu d'apprentissage
ex2_test = ex2[index[(ntrain+1):(ntrain+ntest)],] # création du jeu de test



# ---------- Construction de la foret
### TODO: construire une foret avec 5 arbres.
# Mettre l'argument 'importance' a TRUE lors de la construction:
# randomForest(..., importance=TRUE)




# ---------- Inspection de la foret
###### Question 1:
# (a) Quelle est l'erreur OOB ? 
# (b) Donnez la proportion d'exemples qui n'ont jamais ete utilises pour construire l'un des arbres de la foret ?
# (c) Donnez la proportion  d'exemples qui n'ont jamais fait parti de l'ensemble OOB (pour aucun des arbres de la foret) ?






# ---------- Evaluation de la foret
###### Question 2: Estimez l'erreur en généralisation de cette foret.







# ---------- Selection du meilleur (hyper-)parametre ntree
### TODO: Selectionner la meilleur valeur du parametre ntree
# Aide: utilisez une boucle for : for (val in c(5,10,50,100,500)){...}






###### Question 3: Quelle est l'erreur en generalisation pour le modele selectionne? (Précisez la valeur de l'hyper-parametre selectionne)




# L'importance des predicteurs peut être evaluee (cf. cours, slides 69-70)
nom_model$importance
varImpPlot(nom_model)
###### Question 4: Quels sont les 2 variables les plus importantes ?




### TODO: Calculer combien de fois en moyenne, un exemple fait partie de l'ensemble OOB.




###### Question 5: Ce chiffre vous parait-il coherent ?



# ---------- Comparaison avec un arbre de decision
###TODO: construire un arbre de decision et evaluer ses performances sur l'ensemble de test
###### Question 6: Comparez les résultats du RF avec un arbre de décision sur le meme jeu de données (performances et surfaces de décision).


