

library(randomForest)
library(ROCR)
library(pROC)
library(MASS)
set.seed(11281640)

#Introduce data from Excel to R
mydata=read.csv("~/Desktop/settled.csv")
mydata$INDUSTRY=as.character(mydata$INDUSTRY)
for(i in 1:ncol(mydata))
{
  if(class(mydata[,i])=="factor")
    mydata[,i]=as.numeric(mydata[,i])
}

dele=c(-1,-2,-3,-4,-6,-7,-76,-77,-78,-83,-85)
mydata=mydata[,dele]
mydata=mydata[,-(19:32)]
mydata=mydata[ , -which(names(mydata) %in% c("filed","YEAR","INDUSTRY", "DISMISSED", "ONGOING", "CURR_SCA"))]

##Overall data 
#split the data set into training and testing
set=mydata
size = floor(2 * nrow(set)/3)
split= sample(seq_len(nrow(set)), size = size)  
training =set[split,]
testing =set[-split,]

#build model
rf=randomForest(as.factor(SETTLED)~.,data=training,importance=TRUE,ntree=100)
test_rf=predict(rf,testing,type="prob") 

lg=glm(as.factor(SETTLED)~.,data=training,family="binomial"(link="logit"))
lgp=step(lg,direction="both",test="F")
test_lgp=data.frame(predict(lgp,testing,type="response"))

da=lda(lgp$formula,data = training)
test_da=predict(da,testing)$posterior

pred_rf = prediction(test_rf[,2], testing$SETTLED)
perf_rf = performance(pred_rf, "tpr", "fpr")
auc=performance(pred_rf,"auc")
auc_rf=unlist(slot(auc,"y.values"))

pred_da = prediction(test_da[,2], testing$SETTLED)
perf_da = performance(pred_da, "tpr", "fpr")
auc=performance(pred_da,"auc")
auc_da=unlist(slot(auc,"y.values"))

pred_lgp = prediction(test_lgp[,1], testing$SETTLED)
perf_lgp = performance(pred_lgp, "tpr", "fpr")
auc=performance(pred_lgp,"auc")
auc_lgp=unlist(slot(auc,"y.values"))


plot(perf_da, main="ROC of Settled Frequency ", colorize=T)
plot(perf_da, col=1, add=TRUE)
plot(perf_rf, col=2, add=TRUE)
plot(perf_lgp, col=3, add=TRUE)
legend(0.6, 0.6, c('Discriminant Anlaysis','Random Forest', 'Logistic'), 1:3)

# Save models
final_model=vector("list",3)
final_model[[1]]=rf
final_model[[2]]=lgp
final_model[[3]]=da
save(final_model,file="settled.RData")