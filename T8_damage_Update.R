damage <- read.csv("~/Desktop/TEAM8/damage.csv")
damage_exvars <- names(damage) %in% c("FINAL_DATE", "DISMISSED", "SETTLED","ONGOING","START","END",
                                      "DATE_FILED","TITLE","ID","INDUSTRY","filed","YEAR")
damage <- damage[!damage_exvars]
for(i in 1:ncol(damage))
{
  if(class(damage[,i])=="factor")
    damage[,i]=as.numeric(damage[,i])
}

#1. Divide the data set into training data and testing data （2：1）
set.seed(11281640)
dim(damage)
indexes <- sample(1:nrow(damage), size=(2/3)*nrow(damage))
test <- damage[-indexes,]
dim(test)
train <- damage[indexes,]
dim(train)

#2. Build models 

#(1) linear regression with variable selection
lin_model <- lm(TOTAL~.-P1_MC-P1_ADJCLOSE-P1_STI-P1_REVENUE-P1_PROFIT-P1_MARGIN, data = train)
summary(lin_model)
par(mfrow=c(2,2))
plot(lin_model) # There are influencial points and outliers.
cooks.distance(lin_model)
excooksdist <- which(cooks.distance(lin_model) >= 1)
excooksdist # delete rows with cooks distance >=1
lin_train <- train[-136,] #delete obvious outliers.
lin_train <- lin_train[-264,] 
lin_train <- lin_train[-63,]
lin_train <- lin_train[-45,]
lin_train <- lin_train[-60,]
lin_train <- lin_train[-161,]
lin_train <- lin_train[-450,]
lin_train <- lin_train[-66,]
lin_train <- lin_train[-67,]
lin_train <- lin_train[-360,]
lin_train <- lin_train[-61,]
lin_train <- lin_train[-103,]
lin_train <- lin_train[-105,]
lin_train <- lin_train[-134,]
lin_train <- lin_train[-182,]
lin_train <- lin_train[-187,]
lin_train <- lin_train[-223,]
lin_train <- lin_train[-267,]
lin_train <- lin_train[-294,]
dim(lin_train)
lin_model2 <- lm(TOTAL~.-P1_MC-P1_ADJCLOSE-P1_STI-P1_REVENUE-P1_PROFIT-P1_MARGIN, data = lin_train) #fit linear model using cleaned data
summary(lin_model2)
par(mfrow=c(2,2))
plot(lin_model2)

library(MASS) #Test for transformation
for(i in 1:nrow(lin_train))
{
  if(lin_train[i,"TOTAL"]==0)
    lin_train[i,"TOTAL"]=1
}
a=boxcox(lin_model2)$y
b=boxcox(lin_model2)$x
c=data.frame(a,b)
c[which(c[,1]==max(c[,1])),] # Do not need transformation.

model_both_aic <- step(lin_model2, direction = "both") # Variable selection.
summary(model_both_aic)
lin_pred <- predict(model_both_aic,test)
lin_pred
lin_pred[lin_pred<0] <-0 
lin_pred

#(2) PCA (Principle Component Analysis)
PCA_exvars <- names(train) %in% c("TOTAL") 
PCA_train <- train[!PCA_exvars] # Define the train dataset for PCA.
PCA_test <- test[!PCA_exvars] #Define the test dataset for PCA.
pca_model <- prcomp(PCA_train, scale.=TRUE)
print(pca_model)
plot(pca_model,type="lines") 
summary(pca_model) # Choose PC1-PC15.
pca_model$x[,1:15]
pc_data=data.frame(train$TOTAL,pca_model$x[,1:15])
lm(train.TOTAL~.,data=pc_data)
reg_train<-lm(train.TOTAL~.,data=pc_data)
reg_train1<-step(reg_train,direction = "both")
reg_train2<-lm(train.TOTAL ~ PC1 + PC4 + PC5 + PC7 + PC8 + PC9 + PC11,data=pc_data)
pca_pred <- as.data.frame(predict(pca_model,PCA_test))
pca_pred2 <- predict(reg_train2,pca_pred)
pca_pred2
pca_pred2[pca_pred2<0] <-0
pca_pred2

#(3) neural network
set.seed(11281640)
apply(train,2,function(x) sum(is.na(x))) # No missing value. Good for neural network.
lm.fit <- glm(TOTAL~., data=train)
summary(lm.fit)
pr.lm <- predict(lm.fit,test)
MSE.lm <- sum((pr.lm - test$TOTAL)^2)/nrow(test)
pr.lm

maxs <- apply(train, 2, max)
mins <- apply(train, 2, min)
train_ <- as.data.frame(scale(train, center = mins, scale = maxs - mins))
test_ <- as.data.frame(scale(test, center = mins, scale = maxs - mins))

library(neuralnet)
n <- names(train_)
f <- as.formula(paste("TOTAL ~", paste(n[!n %in% "TOTAL"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(37,25,17,11,7,5,3,2),linear.output=T)

dim(test_)
pr.nn <- compute(nn,test_[,1:54])
pr.nn_ <- pr.nn$net.result*(max(damage$TOTAL)-min(damage$TOTAL))+min(damage$TOTAL)
test.r <- (test_$TOTAL)*(max(damage$TOTAL)-min(damage$TOTAL))+min(damage$TOTAL)
MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)
print(paste(MSE.lm,MSE.nn))
pr.nn_

#3.MAPE as selection criteria 
lin_mape <- mean(abs(test$TOTAL-lin_pred)/test$TOTAL)
lin_mape

pca_mape <- mean(abs(test$TOTAL-pca_pred2)/test$TOTAL)
pca_mape

nn_mape <- mean(abs(test$TOTAL-pr.nn_)/test$TOTAL)
nn_mape
# Linear regression and PCA have similarly low MAPEs. The final prediction result takes the average of the two.

# Save model
damage_model=vector("list",3)
damage_model[[1]]=model_both_aic
damage_model[[2]]=pca_model
damage_model[[3]]=reg_train2
save(damage_model,file="damage.RData")

