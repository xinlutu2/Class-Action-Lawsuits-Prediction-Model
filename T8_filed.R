total=read.csv("~/Desktop/filed.csv")
total$INDUSTRY=as.character(total$INDUSTRY)
industry=read.csv("~/Desktop/Industry.csv",header=F)
industry$V1=as.character(industry$V1)

library(randomForest)
library(e1071)
library(ROCR)
library(pROC)
library(caret)
library(MASS)
set.seed(11281640)

##### for overall data 
set=total

#split the data set into training and testing
size = floor(2 * nrow(set)/3)
split= sample(seq_len(nrow(set)), size = size)  
training =set[split,]
testing =set[-split,]

#build model 

#logistic model
#lg=glm(as.factor(FILED)~.-YEAR,data=training[,!colnames(training) %in% c("INDUSTRY","ID","TITLE")],family='binomial'(link='logit'))
#step(lg,direction="both",test="F")
# Since there are more than 30,000 observtions in the data set, the stepwise selection is very slow. Thus,
# we directly put the resulting model below.
lg=glm(formula = as.factor(FILED) ~ ADJ_CLOSE + M_CAP + M_CAP_MID + 
         NI + OI + REVENUE + PROFIT + MARGIN + ASSETS + DEBT + EQUITY + 
         DE + STI + INTEREST + IC + PRIOR_SCA + CURR_SCA + P_ADJ_CLOSE + 
         P_M_CAP + P_NI + P_OI + P_REVENUE + P_PROFIT + P_MARGIN + 
         P_ASSETS + P_DEBT + P_EQUITY + P_DE + P_STI + P_INTEREST + 
         P_IC + Z_SCORE + P1_MC + P1_MC_MID + P2_MC + P2_MC_MID + 
         P3_MC + P3_MC_MID + P1_ADJCLOSE + P2_ADJCLOSE + P3_ADJCLOSE + 
         P2_STI + P3_STI + P1_PROFIT + P2_REVENUE + P2_PROFIT + P2_MARGIN + 
         P3_REVENUE + P3_PROFIT + P3_MARGIN, family = binomial(link = "logit"), 
       data = training[, !colnames(training) %in% c("INDUSTRY", 
                                                    "ID", "TITLE")])
test_lg=predict(lg,testing,type='response')
#discriminant anlaysis
da=lda(lg$formula,data = training[, !colnames(training) %in% c("INDUSTRY","ID", "TITLE")])
test_da=predict(da,testing)$posterior
#random forest
rf=randomForest(as.factor(FILED)~.-INDUSTRY-ID-YEAR-TITLE,data=training,importance=TRUE,ntree=100)
test_rf=predict(rf,testing,type="prob")

# ROC curve
pred_lg = prediction(test_lg, testing$FILED)
perf_lg = performance(pred_lg, "tpr", "fpr")
auc=performance(pred_lg,"auc")
auc_lg=unlist(slot(auc,"y.values"))

pred_da = prediction(test_da[,2], testing$FILED)
perf_da = performance(pred_da, "tpr", "fpr")
auc=performance(pred_da,"auc")
auc_da=unlist(slot(auc,"y.values"))

pred_rf = prediction(test_rf[,2], testing$FILED)
perf_rf = performance(pred_rf, "tpr", "fpr")
auc=performance(pred_rf,"auc")
auc_rf=unlist(slot(auc,"y.values"))

plot(perf_lg, main="ROC of Filed Frequency", colorize=T)
plot(perf_lg, col=1, add=TRUE)
plot(perf_da, col=2, add=TRUE)
plot(perf_rf, col=3, add=TRUE)
legend(0.6,0.4, legend=c('Logistic','Discrimant Analysis','Random Forest'), 1:3,cex=0.75)

# Select RandomForest  and LDA as the overall model
overallmodel=vector("list",2)
overallmodel[[1]]=da
overallmodel[[2]]=rf
save(overallmodel,file="filed_all.RData")

##### for each industry

# Attention: due to the stepwise function, the following for loop will be very slow
#            and sometimes errors may occur. However, it did pass all the code if we
#            manually restart it again from where it failed before. We have successfully
#            run all the codes and save the models(filed_industry.RData) into a vector. 
#            Thus, we can directly use those models without rerun this code again. 

models=vector("list",64)
for(i in 1:length(industry[,1])){
  set=total[which(total$INDUSTRY==industry[i,1]),]
  
  if(sum((set$FILED))!=0){ # no filed case, can not predict
    
    #split the data set into training and testing
    size = floor(2 * nrow(set)/3)
    split= sample(seq_len(nrow(set)), size = size)  
    training =set[split,]
    testing =set[-split,]
    
    lg=glm(as.factor(FILED)~.-YEAR,data=training[,!colnames(training) %in% c("INDUSTRY","ID","TITLE")],family='binomial'(link='logit'))
    lg=step(lg,direction="both",test="F")
    test_lg=predict(lg,testing,type="response")
    da=lda(lg$formula,data=training[,!colnames(training) %in% c("INDUSTRY","ID","TITLE")])
    test_da=predict(da,testing)$posterior
    rf=randomForest(as.factor(FILED)~.-INDUSTRY-ID-YEAR-TITLE,data=training,importance=TRUE,ntree=100)
    test_rf=predict(rf,testing,type="prob")
    
    #prepare model for ROC Curve
    if(sum(testing$FILED)>1){ # ROC cannot plot if no instance is in the test data set
      pred_rf = prediction(test_rf[,2], testing$FILED)
      perf_rf = performance(pred_rf, "tpr", "fpr")
      auc=performance(pred_rf,"auc")
      auc_rf=unlist(slot(auc,"y.values"))
      
      pred_lg = prediction(test_lg, testing$FILED)
      perf_lg = performance(pred_lg, "tpr", "fpr")
      auc=performance(pred_lg,"auc")
      auc_lg=unlist(slot(auc,"y.values"))
      
      pred_da = prediction(test_da[,2], testing$FILED)
      perf_da = performance(pred_da, "tpr", "fpr")
      auc=performance(pred_da,"auc")
      auc_da=unlist(slot(auc,"y.values"))
      
      plot(perf_lg, main="ROC", colorize=T)
      plot(perf_lg, col=1, add=TRUE)
      plot(perf_da, col=2, add=TRUE)
      plot(perf_rf, col=3, add=TRUE)
      legend(0.6, 0.6, c('Logistic','Discrimant Analysis','Random Forest'), 1:3)
      
      # select model by comparing auc_xx
      if (auc_rf>auc_da){
        if(auc_rf>auc_lg) models[[i]]=rf
        else models[[i]]=lg
      } else if(auc_da>auc_lg) models[[i]]=da else models[[i]]=lg
      
    }
  }
}

# Save models
save(models,file="filed_industry.RData")
