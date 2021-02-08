# Fonctions dessinant les frontieres d'un modele dans le plan (les données doivent être en 2 dimensions)..
# Les données d'apprentissage et de validation sont également représentées dans le plan.
# Les extrémités du plan sont données par xmin, xmax, ymin, ymax. 
# Pour tous les points du plan, on prédit, en utilisant le modèle, quelle est la classe de ce point, 
# et on l'affiche dans le plan avec la couleur associée (coul0 si classe 0, coul1 si classe 1).

dessiner_frontiere_tree = function(data_app, data_val,tree, xm,xM,ym,yM, cols){
	
	dev.new()
	y = data_app[,ncol(data_app)]
	y2 = data_val[, ncol(data_val)]
	if(length(table(y))==2){
	plot(data_app[which(y==0),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
	points(data_app[which(y==1),1:2], col = cols[2])
	points(data_val[which(y2==0),1:2], col = cols[1], pch = 2)
	points(data_val[which(y2==1),1:2], col = cols[2], pch = 2)
	}
	else{
		plot(data_app[which(y==1),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
		points(data_val[which(y2==1),1:2], col = cols[1], pch = 2)
		for(j in 2:length(table(y))){
			
			points(data_app[which(y==j),1:2], col = cols[j])
			points(data_val[which(y2==j),1:2], col = cols[j], pch = 2)
		}
		
	}	
	
	x1 = seq(xm,xM, length.out = 60)
	x2 = seq(ym, yM, length.out = 60)
	
	for(i in 1:60){
			for(j in 1:60){
				ii = as.data.frame(rbind(c(x1[i],x2[j])))
				colnames(ii) = colnames(data_app[,-ncol(data_app)])
				p= predict(tree,ii, type = "class")
				if(length(table(y))==2){
				  coul = ifelse(p==0,1,2) 
				  points(x1[i],x2[j], col = cols[coul], pch = 3)}
				else{
						points(x1[i],x2[j], col = cols[p], pch = 3)}

				}
			}	
}


dessiner_marge = function(X, model, xm,xM,ym,yM, cols){
  dev.new()
  SV = X[model$index,1:2]
  SV_y= X[model$index,3]
  X2 = X[-model$index,1:2]
  y2= X[-model$index,3]
  
  if(length(table(X$y))==2){
    plot(X2[which(y2==0),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
    points(X2[which(y2==1),1:2], col = cols[2])
    points(SV[which(SV_y==0),1:2], col = cols[1], pch= 13)
    points(SV[which(SV_y==1),1:2], col = cols[2], pch= 13)
  }
  else{
    plot(X[which(y==1),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
    for(j in 2:length(table(y))){
      points(X[which(y==j),1:2], col = cols[j])
    }
  }
  
  w = t(model$coefs)%*%model$S
  b = model$rho
  abline(b/w[2], -w[1]/w[2])
  abline((b+1)/w[2],-w[1]/w[2],lty=2)
  abline((b-1)/w[2],-w[1]/w[2],lty=2)
}


dessiner_frontiere_svm = function(X,y, Xvalid, yvalid, model, xm,xM,ym,yM, cols){
  
  dev.new()
  SV = X[model$index,]
  SV_y= y[model$index]
  X2 = X[-model$index,]
  y2= y[-model$index]
  
  if(length(table(y))==2){
    plot(X2[which(y2==0),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
    points(X2[which(y2==1),1:2], col = cols[2])
    points(SV[which(SV_y==0),1:2], col = cols[1], pch= 13)
    points(SV[which(SV_y==1),1:2], col = cols[2], pch= 13)
    points(Xvalid[which(yvalid==0),1:2], col = cols[1], pch = 2)
    points(Xvalid[which(yvalid==1),1:2], col = cols[2], pch = 2)
  }
  else{
    plot(X[which(y==1),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
    for(j in 2:length(table(y))){
      points(X[which(y==j),1:2], col = cols[j])
    }
  }	
  
  x1 = seq(xm,xM, length.out = 60)
  x2 = seq(ym, yM, length.out = 60)
  
  for(i in 1:60){
    for(j in 1:60){
      p= predict(model, rbind(c(x1[i],x2[j])))
      if(length(table(y))==2){
        if(p==0){points(x1[i],x2[j], col = cols[1], pch = 3)}
        else{points(x1[i],x2[j], col = cols[2], pch = 3)}
      }
      else{
        points(x1[i],x2[j], col = cols[p], pch = 3)}
    }
  }	
}


dessiner_frontiere_NN = function(X,y,model, xm,xM,ym,yM, cols){
  
  dev.new()
  if(length(table(y))==2){
    plot(X[which(y==0),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
    points(X[which(y==1),1:2], col = cols[2])
  }
  else{
    plot(X[which(y==1),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
    for(j in 2:length(table(y))){
      
      points(X[which(y==j),1:2], col = cols[j])
      
    }
    
  }	
  nb = 60
  x1 = seq(xm,xM, length.out = nb)
  x2 = seq(ym, yM, length.out = nb)
  
  for(i in 1:nb){
    for(j in 1:nb){
      p= ifelse(predict(model, rbind(c(x1[i],x2[j])))>0.5,1,0)
      if(length(table(y))==2){
        if(p==0){points(x1[i],x2[j], col = cols[1], pch = 3)}
        else{points(x1[i],x2[j], col = cols[2], pch = 3)}
      }
      else{
        points(x1[i],x2[j], col = cols[p], pch = 3)}
    }
  }	
  
}


plot_NN_loss=function(history){
  dev.new()
  # Cas ou il n'y a pas d'ensemble de validation: 
  # is.vector(history$metrics$val_loss) == FLASE ou length(history$metrics$val_loss)==0
  y_max=ifelse(length(history$metrics$val_loss)==0, max(history$metrics$loss), max(max(history$metrics$loss),max(history$metrics$val_loss)))+0.05
  #y_max=max(history$metrics$val_loss)+0.05
  # Plot the model loss of the training data
  plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l",ylim=c(0, y_max))
  # Plot the model loss of the test data
  lines(history$metrics$val_loss, col="green")
  # Add legend
  legend("topright", c("train","val"), col=c("blue", "green"), lty=c(1,1))
}


plot_NN_accuracy=function(history){
  dev.new()
  plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l", ylim=c(0, 1))
  lines(history$metrics$val_acc, col="green")
  legend("bottomright", c("train","val"), col=c("blue", "green"), lty=c(1,1))
}





dessiner_frontiere_foret= function(data_app, data_val,foret, xm,xM,ym,yM, cols){
	
	dev.new()
	y = data_app[,ncol(data_app)]
	y2 = data_val[, ncol(data_val)]
	if(length(table(y))==2){
	plot(data_app[which(y==0),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
	points(data_app[which(y==1),1:2], col = cols[2])
	points(data_val[which(y2==0),1:2], col = cols[1], pch = 2)
	points(data_val[which(y2==1),1:2], col = cols[2], pch = 2)
	}
	else{
		plot(data_app[which(y==1),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
		points(data_val[which(y2==1),1:2], col = cols[1], pch = 2)
		for(j in 2:length(table(y))){
			
			points(data_app[which(y==j),1:2], col = cols[j])
			points(data_val[which(y2==j),1:2], col = cols[j], pch = 2)
		}
		
	}	
	
	x1 = seq(xm,xM, length.out = 60)
	x2 = seq(ym, yM, length.out = 60)
	
	for(i in 1:60){
			for(j in 1:60){
				ii = as.data.frame(rbind(c(x1[i],x2[j])))
				colnames(ii) = colnames(data_app[,-ncol(data_app)])
				p= predict(foret,ii)
				if(length(table(y))==2){
				  coul = ifelse(p==0,1,2) 
				  points(x1[i],x2[j], col = cols[coul], pch = 3)}
				else{
					points(x1[i],x2[j], col = cols[p], pch = 3)}
				}
			}	
}


dessiner_frontiere_knn= function(data_app, data_val, k, xm,xM,ym,yM, cols){
  
  dev.new()
  y = data_app[,ncol(data_app)]
  y2 = data_val[, ncol(data_val)]
  if(length(table(y))==2){
    plot(data_app[which(y==0),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
    points(data_app[which(y==1),1:2], col = cols[2])
    points(data_val[which(y2==0),1:2], col = cols[1], pch = 2)
    points(data_val[which(y2==1),1:2], col = cols[2], pch = 2)
  }
  else{
    plot(data_app[which(y==1),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
    points(data_val[which(y2==1),1:2], col = cols[1], pch = 2)
    for(j in 2:length(table(y))){
      
      points(data_app[which(y==j),1:2], col = cols[j])
      points(data_val[which(y2==j),1:2], col = cols[j], pch = 2)
    }
    
  }	
  
  x1 = seq(xm,xM, length.out = 60)
  x2 = seq(ym, yM, length.out = 60)
  
  for(i in 1:60){
    for(j in 1:60){
      ii = as.data.frame(rbind(c(x1[i],x2[j])))
      colnames(ii) = colnames(data_app[,-ncol(data_app)])
      p= knn(data_app[,1:2], ii, data_app[,3], k)
      if(length(table(y))==2){
        coul = ifelse(p==0,1,2) 
        points(x1[i],x2[j], col = cols[coul], pch = 3)}
      else{
        points(x1[i],x2[j], col = cols[p], pch = 3)}
    }
  }	
}





dessiner_frontiere_reglog = function(data_app,data_val,modele, puissance, coul0, coul1, xmin, xmax,ymin, ymax){
  
  # Le paramètre puissance représente la puissance à laquelle on a élévé les données (par la fonction polynomial).
  # puissance = 1 si on n'a rien fait de particulier 
  dev.new()
  plot(data_app[which(data_app[,ncol(data_app)]==0),1:2], col = coul0, xlim = c(xmin,xmax), ylim = c(ymin,ymax))
  points(data_app[which(data_app[,ncol(data_app)]==1),1:2], col = coul1, xlim = c(xmin,xmax), ylim = c(ymin,ymax))
  
  points(data_val[which(data_val[,ncol(data_val)]==1),1:2], col = coul1, pch = 2)
  points(data_val[which(data_val[,ncol(data_val)]==0),1:2], col = coul0, pch = 2)
  
  
  x = seq(xmin,xmax, length.out = 50)
  y = seq(ymin, ymax, length.out = 50)
  
  if(puissance ==1){
    for(i in 1:50){
      for(j in 1:50){
        new=as.data.frame(rbind(c(x[i],y[j])))
        colnames(new) = colnames(data_app)[1:(ncol(data_val)-1)]
        p= predict(modele, new, typ = 'response')
        col = ifelse(p>=0.5,1,0) 
        if(col==0){points(x[i],y[j], col = coul0, pch = 3)}
        else{points(x[i],y[j], col = coul1, pch = 3)}
        
      }
    }	
    
  }
  else{
    
    for(i in 1:50){
      for(j in 1:50){
        new=as.data.frame(rbind(c(x[i],y[j],0)));
        colnames(new) = colnames(data_val)[1:(ncol(data_val))]
        new_p= polynomial(new,puissance);
        p = predict(modele,new_p[,1:(ncol(new_p)-1)], typ = 'response')
        col = ifelse(p>=0.5,1,0) 
        if(col==0){points(x[i],y[j], col = coul0, pch = 3)}
        else{points(x[i],y[j], col = coul1, pch = 3)}
        
      }
    }
  }
}


# Pour la regression logistique
polynomial = function(data, puissance){
  
  # Cette fonction élève les données (les 2 coordonnées x1 et x2) à la puissance p en créant les nouvelles variables :
  # x1^p, x2^p, x1*x2^(p-1), x1^(p-1)*x2, x2^2*x1^(p-2), x1^(p-2)*x2^2, etc...
  # on crée une matrice à 0 colonnes (pour l instant)
  new_data = matrix(0,nrow(data),0) 
  
  for(i in 1:puissance){
    for(j in 0:i){
      new_data = cbind(new_data,data[,1]^(i-j)*data[,2]^(j))
    }
  }
  
  # on met la classe en dernière colonne
  new_data = cbind(new_data, data[,ncol(data)])
  colnames(new_data) = c(paste("V", c(1:(ncol(new_data)-1)), sep=""),colnames(data)[ncol(data)])
  
  return(as.data.frame(new_data))
}

