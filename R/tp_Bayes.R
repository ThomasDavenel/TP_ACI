# ---------- Chargement des packages nécéssaires
library(e1071)
source("fonctions_utiles.R")


# ---------- Chargement des données 
ex1 = read.table("./data/data_exam.txt", header = T)
ex1$Reussite = as.factor(ex1$Reussite)
str(ex1)
table(ex1$Reussite)


# ---------- Separation Apprentissage/Test
nall = nrow(ex1) #total number of rows in data
ntrain = floor(0.7 * nall) # number of examples (rows) for train: 70% (vous pouvez changer en fonction des besoins)
ntest = nall - ntrain # number of examples for test: le reste
set.seed(20) # choix d'une graine pour le tirage aléatoire
index = sample(nall) # permutation aléatoire des nombres 1, 2, 3 , ... nall
ex1_app = ex1[index[1:ntrain],] # création du jeu d'apprentissage
ex1_test = ex1[index[(ntrain+1):(ntrain+ntest)],] # création du jeu de test

table(ex1_app$Reussite)
table(ex1_test$Reussite)


# ----- Visualisation des données apprentissage et validation
#Affichage des données d apprentissage
plot(ex1_app[which(ex1_app$Reussite==0),1:2], col = "red", xlim = c(0,105), ylim = c(0,105))
points(ex1_app[which(ex1_app$Reussite==1),1:2], col = "blue")
# Affichage des données de test (avec un triangle , pch = 2)
points(ex1_test[which(ex1_test$Reussite==1),1:2], col = "blue", pch = 2)
points(ex1_test[which(ex1_test$Reussite==0),1:2], col = "red", pch = 2)



# ---------- Construction du classifieur Bayesien Naif
# Pour le bayesien naif, il n y a pas de paramètres à ajuster donc on apprend le modèle sur
# l'ensemble d'apprentissage uniquement (pas d'ensemble de validation)
bayes = naiveBayes(Reussite~., data = ex1_app)

# Vous pouvez voir les probabilites a priori, ainsi que les paramètres des gaussiennes estimées en tapant le nom du modèle :
print(bayes$apriori)
print(bayes$tables)
# ou print(bayes)

# A-priori probabilities:
# Y
# 		0         1 
# 0.3882353 0.6117647 

# Conditional probabilities:
# moyenne et ecart-type sachant Y dans les 2 colonnes des matrices données à l'écran

   # Note1 = x1
# Y       [,1]     [,2]
  # 0 53.07455 17.69572	-> moyenne [,1] et ecart-type [,2] de x1|C1
  # 1 73.91385 14.68127	-> moyenne et ecart-type de x1|C2

   # Note2 = x2
# Y       [,1]     [,2]
  # 0 53.12424 15.36159
  # 1 74.77731 16.17977


#### Visualisation des vraisemblances
# Affichage des densites de probabilités pour la variable x1=Note1
note1_c1=as.numeric(unlist(ex1_app[ex1_app$Reussite==0,][1])) #conversion to numeric for density plot
note1_c2=as.numeric(unlist(ex1_app[ex1_app$Reussite==1,][1])) 
# Pour la classe 0
plot(density(note1_c1),lty=2,col="red", xlab="Note 1", main="Density estimation for Note1") # estimation par KDE
points(note1_c1, y= rep(0.00,length(note1_c1)),col="red")  # affichage des points de l'ensemble d'apprentissage
curve(dnorm(x, bayes$tables$Note1[1,1], bayes$tables$Note1[1,2]), add=TRUE, col="red") # gaussienne estimée par MV
# Idem pour la classe 1
lines(density(note1_c2),lty=2,col="blue")
points(note1_c2, y= rep(0.00,length(note1_c2)),col="blue")
curve(dnorm(x, bayes$tables$Note1[2,1], bayes$tables$Note1[2,2]), add=TRUE, col="blue")
legend("topright", c("KDE class 0", "gaussian class 0","KDE class 1", "gaussian class 1"), col = c("red","red","blue","blue"), lty=c(2,1,2,1), cex=0.5)



# ---------- Prediction avec Bayes

###### Question 1: Calculer la classe du premier exemple de l'ensemble de test. Donnez les valeurs numériques calculées.
# Aide:
# - dnorm(x, mean = ??, sd = ??) calcul la densité de proba au point x 
# d'une distribution gaussienne de moyenne 'mean' et d'ecart-type 'sd'
# - bayes$tables$Note1[1,1] et bayes$tables$Note1[1,2] pour accéder aux paramètres appris



# La fonction predict marche avec Bayes
# Vous pouvez prédire la classe de tous les individus de l'ensemble de test d'un coup:
predict(bayes, ex1_test)


# ---------- Affichage de la frontière de décision
# idem tree
dessiner_frontiere_tree(ex1_app, ex1_test, bayes, 0,105,0,105,c("red", "blue"))

###### Question 2: Estimez l'erreur de généralisation faite par bayes sur ces données.







