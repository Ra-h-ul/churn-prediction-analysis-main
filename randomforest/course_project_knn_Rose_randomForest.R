library('class')
library('smotefamily')
library('caret')
library('randomForest')
library('dplyr')
library('ROSE')

library('VIM')     #for.KNN.imputation 

#reading csv file -------
data<-read.csv('Churn_Modelling.csv')

#converting from int to factor ----------
data$Exited<-as.factor(data$Exited)
data$IsActiveMember<-as.factor(data$IsActiveMember)
data$HasCrCard<-as.factor(data$HasCrCard)



#converting 0 balance to NA ---------

data$Balance[data$Balance == 0 ] = NA
# data.numeric <- select_if(data, is.numeric)


# randomly reorder the data-------

set.seed(1234)                                        # setting a seed for sample function
new.data<-data[sample(1:nrow(data)),]

#removing outliers --------
new.data.modified<-subset(new.data,new.data$CreditScore>405 & Age<65)

#KNN imputation-----
new.knn.data<-new.data.modified
new.knn.imputed<-kNN(new.knn.data , variable= c('Balance') , k=7)
new.knn.imputed<-subset(new.knn.imputed , select = RowNumber : Exited)     


#train test-------
set.seed(12345)
sample.size<-sample(2,nrow(new.knn.imputed),replace=TRUE , prob = c(0.7,0.3))
train<- new.knn.imputed[sample.size==1,]
test<- new.knn.imputed[sample.size==2,]

#checking class imbalance--
summary(train$Exited)



#correcting class imbalance ---------
over<-ovun.sample(Exited~.,data=train , method = 'over' , N=10826)$data
under<-ovun.sample(Exited~.,data=train , method = 'under' , N= 2730)$data
both<-ovun.sample(Exited~.,data=train , method = 'both' , 
                  p=0.5,  seed=123,  N=5431)$data


#model----------
rf.train <-randomForest(Exited~.,data=train)
rf.over <-randomForest(Exited~.,data=over)
rf.under <-randomForest(Exited~.,data=under)
rf.both <-randomForest(Exited~.,data=both)



#evaluation------

df1<-table(predict(rf.train,test),test$Exited)
prec<- df1[1,1]/sum(df1[1,1],df1[2,1])
rec<- df1[1,1]/sum(df1[1,1],df1[1,2])

confusionMatrix(predict(rf.train,test),test$Exited ) #Accuracy :   Sensitivity :   Specificity :    
cat("\n precision",prec)               
cat("\n recall ",rec)                  
cat("\n f-score:",2*(prec*rec)/(prec+rec))   




df2<-table(predict(rf.over,test),test$Exited)
prec2<- df2[1,1]/sum(df2[1,1],df2[2,1])
rec2<- df2[1,1]/sum(df2[1,1],df2[1,2])

confusionMatrix(predict(rf.over,test),test$Exited )  #Accuracy :  Sensitivity : Specificity : 
df2<-table(predict(rf.over,test),test$Exited)
cat("\n precision",prec2)               
cat("\n recall ",rec2)                 
cat("\n f-score:",2*(prec2*rec2)/(prec2+rec2))   





df3<-table(predict(rf.under,test),test$Exited)
prec3<- df3[1,1]/sum(df3[1,1],df3[2,1])
rec3<- df3[1,1]/sum(df3[1,1],df3[1,2])

confusionMatrix(predict(rf.under,test),test$Exited )  #Accuracy :   Sensitivity :  Specificity :   
cat("\n precision",prec3)                
cat("\n recall ",rec3)                  
cat("\n f-score:",2*(prec3*rec3)/(prec3+rec3))   







df4<-table(predict(rf.both,test),test$Exited)
prec4<- df4[1,1]/sum(df4[1,1],df4[2,1])
rec4<- df4[1,1]/sum(df4[1,1],df4[1,2])


confusionMatrix(predict(rf.both,test),test$Exited )   #Accuracy :   Sensitivity :  Specificity : 
cat("\n precision",prec4)                
cat("\n recall ",rec4)                   
cat("\n f-score:",2*(prec4*rec4)/(prec4+rec4))   

