# ---------- Chargement des packages nécéssaires
library(rpart)
install.packages("rpart.plot")
library(rpart.plot)
source("fonctions_utiles.R")


########
# Load and analyze dataset
########

# ---------- Chargement des données
ex1 = read.table("./data/data_exam.txt", header = T)
# some machine learning methods (trees, bayes, knn...) require the class to be of type 'factor' to work.
# Other don't (SVM, logistic regression, NN).
ex1$Reussite = as.factor(ex1$Reussite) # convert the class in 'factor' type, needed for datasets which class is not a factor


# The first mandatory step is to look at your data
# Question: You always should be able to say:
# - How many examples in the dataset?
# - How many features? What are their types? Their distribution?
# - What are the classes?
# - What is the class distribution over the dataset?
head(ex1)
str(ex1)
table(ex1$Reussite)




# ---------- Separation Apprentissage/Validation/Test
ex1_app = ex1[1:70,] # 70% donnees apprentissage
ex1_val = ex1[71:85,] # 15% donnees validation
ex1_test = ex1[86:100,] # 15% donnees test


# ----- Visualisation des données apprentissage et validation

#Affichage des données d apprentissage
# D'abord ceux qui ont échoués (pour lesquels ex1$Reussite = 0) en rouge
plot(ex1_app[which(ex1_app$Reussite==0),1:2], col = "red", xlim = c(0,105), ylim = c(0,105))
# puis ceux qui ont réussis, en bleu
points(ex1_app[which(ex1_app$Reussite==1),1:2], col = "blue")

# Affichage des données de validation (on les affiche avec un triangle , pch = 2)
points(ex1_val[which(ex1_val$Reussite==1),1:2], col = "blue", pch = 2)
points(ex1_val[which(ex1_val$Reussite==0),1:2], col = "red", pch = 2)



# ---------- Construction de l'arbre

# La fonction rpart permet d'ajuster un arbre de décision à partir d'un ensemble d'apprentissage.
# Les paramètres sont :
# - une formule  : NomdeColonne~., data = ... NomDeColonne correspond au nom de la colonne qui contient la classe
# dans les données (ici Reussite), et on doit mettre le nom de la variable qui contient les données après data = 
# - control = list(minbucket = 1,cp = 0, minsplit = 1) ces paramètres signifient que l'on souhaite construire l'arbre entièrement (sans aucun élagage). On verra tout à l'heure comment élaguer
tr = rpart(Reussite~., data = ex1_app, control = list(minbucket = 1,cp = 0, minsplit = 1))


# ---------- Visualisation de l'arbre
rpart.plot(tr, extra = 1)
# A chaque noeud (et feuille) de l'arbre, vous retrouvez plusieurs informations
# - la classe majoritaire (ici 0 ou 1) dans ce noeud  
# - le nombre d'individus de la classe 0 (de l'ensemble d'apprentissage) qui sont affectés à cette feuille
# - le nombre d'individus de la classe 1 (de l'ensemble d'apprentissage) qui sont affectés à cette feuille
# - la question que l'on va poser dans ce noeud (pas de questions si on est dans une feuille). On part à gauche, si réponse est "oui", et à droite sinon.




# ---------- Prédiction à la main de la classe d'un individu 

# Regardez le premier individu de l'ensemble d'apprentissage :
ex1_app[1,]
# Parcourir l'arbre pour cet individu jusqu'à arriver à une feuille. 
# Question : Quelle classe prédisez-vous pour cet individu ? L'arbre se trompe t'il pour cet individu ?
# Essayez avec 1 ou 2 autres individus au choix. L'arbre se trompe t'il ? Est-ce normal ?


# La fonction predict permet de faire automatiquement le cheminement que vous venez de faire à la main:
# pour le premier individu de l'ensemble d'apprentissage
predict(tr, ex1_app[1,], type = 'class')
# Verifiez que la classe prédite est bien la même que ce que vous avez fait à la main
# Vous pouvez prédire la classe de tous les individus de l'ensemble d'apprentissage d'un coup:
pa=predict(tr, ex1_app, type = "class")
print(pa)



# Question : Calculer le nombre de prédictions correctes faite par cet arbre sur l'ensemble d'apprentissage
# Aide : commande sum(predict(....)==...)
predictionCorecte = sum(predict(tr, ex1_app, type = "class")==ex1_app[,3])
print(predictionCorecte)


# matrice de confusion
CMa1=table(ex1_app$Reussite, pa,dnn=list('actual','predicted'))
print(CMa1)

# Question : Idem pour les individus de l'ensemble de test Quelles differences observez-vous entre ces 2 jeux de donnees ?







# ---------- Affichage de la frontière de décision

# Dans le fichier "fonctions_utiles.R", il y a une fonction qui permet de dessiner la frontière de décision associée à un arbre de décision (dessiner_frontiere_tree)
# Le principe : prédire tous les points du plan et les afficher avec la couleur associée à la prédiction.
# Les données d'apprentissage et de validation sont également affichées.
# Les paramètres sont (dans l'ordre): 
# - l ensemble d apprentissage
# - l'ensemble de validation ou de test
# - le modèle : ici l'arbre en question
# - coordoonnées xm puis xM : pour donner l'intervalle de tracé sur l'axe des abscisses ici 0 et 105
# - coordoonnées ym puis yM : pour donner l'intervalle de tracé sur l'axe des ordonnées ici 0 et 105
# - un vecteur qui donne les couleurs avec lesquelles on veut tracer les classes, c("red", "blue") par exemple
dessiner_frontiere_tree(ex1_app, ex1_val, tr, 0,105,0,105,c("red", "blue"))

# Question : La forme de la frontière vous semble t'elle bien correspondre à un arbre de décision ?
# Question : Sur le graphe, essayer de visualiser les points de l'ensemble de validation (dessinés avec un triangle) qui sont mal classés par l'arbre.




# ---------- Elagage de l'arbre 

# Avec R, l'élagage de l'arbre se fait via le paramètre cp utilisé dans l'appel à la fonction rpart.
# Tout à l'heure, nous avions mis cp = 0, ce qui correspond à l'arbre complet. cp correpond à un paramètre
# de complexité (complexity parameter). Plus on élague l'arbre moins il est complexe, mais par contre ses performances sur l'ensemble d'apprentissage seront moins bonnes.
# En augmentant cp petit à petit, on élague l'arbre, et on cherche un compromis entre complexité et performance.
# Les valeurs de cp permettant d'élaguer l'arbre peuvent être trouvées en regardant le résultat de la commande suivante : 
rev(tr$cptable[,1])
# Cette commande vous donne toutes les valeurs possibles de cp qui permettent d'élaguer l'arbre de plus en plus.
# L'élagage se fait ensuite via la commande prune : 
tr_elague = prune(tr, cp = 0) # si on met cp = 0, on ne fait aucun élagage. Vous pouvez vérifier en affichant tr_elague:
rpart.plot(tr_elague, extra = 1) # Cela donne le même arbre que tr


#Question : Pour chacune des autres valeurs de cp (données par rev(tr$cptable[,1]))
#				- afficher l'arbre élagué
#				- calculer le nombre de bonnes prédictions de cet arbre sur l'ensemble d'apprentissage
#				- calculer le nombre de bonnes prédictions de cet arbre sur l'ensemble de validation
# Aide: utilisez une boucle for : for (val in  rev(tr$cptable[,1])){...}
for (val in  rev(tr$cptable[,1])){
  print("valeur de cp :")
  print(val)
  tr_elague = prune(tr, cp = val)
  rpart.plot(tr_elague, extra = 1)
  predictionCorecte = sum(predict(tr_elague, ex1_app, type = "class")==ex1_app[,3])
  print("nombre de bonnes pr�dictions de cet arbre sur l'ensemble d'apprentissage :")
  print(predictionCorecte)
  print(nrow(ex1_app))
  print("")
  print("nombre de bonnes pr�dictions de cet arbre sur l'ensemble de validation :")
  predictionCorecte = sum(predict(tr_elague, ex1_val, type = "class")==ex1_val[,3])
  print(predictionCorecte)
  print(nrow(ex1_val))
  print("")
}

# Question : Quel arbre choisissez-vous de garder pour ces données ?

#-> cp =0.04


# Question : Estimez l'erreur de généralisation faite par cet arbre.
tr_elague = prune(tr, cp = 0.04) 
rpart.plot(tr_elague, extra = 1)
print("Erreur de Generalisation :")
predictionTeste = sum(predict(tr_elague, ex1_test, type = "class")==ex1_test[,3])
print(predictionTeste)
print(nrow(ex1_test))

# Question: compute the per-class precision and recall this tree. Which class is classified the best?

for (val in  rev(tr$cptable[,1])){
  print("valeur de cp :")
  print(val)
  tr_elague = prune(tr, cp = val) 
  rpart.plot(tr_elague, extra = 1)
  print("Erreur de Generalisation :")
  predictionTeste = sum(predict(tr_elague, ex1_test, type = "class")==ex1_test[,3])
  print(predictionTeste)
  print(nrow(ex1_test))
}

