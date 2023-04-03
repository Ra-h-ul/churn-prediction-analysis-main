library('class')
library('smotefamily')
library('caret')
library('randomForest')
library('dplyr')
library('ROSE')
library('e1071')

library('VIM')     #for.KNN.imputation 

#reading csv file -------
data<-read.csv('Churn_Modelling.csv')

#converting from int to factor ----------
data$Exited<-as.factor(data$Exited)
data$IsActiveMember<-as.factor(data$IsActiveMember)
data$HasCrCard<-as.factor(data$HasCrCard)



#converting 0 balance to NA ---------

data$Balance[data$Balance == 0 ] = NA
data.numeric <- select_if(data, is.numeric)


# randomly reorder the data-------

set.seed(1234)                                        # setting a seed for sample function
new.data<-data[sample(1:nrow(data)),]


#removing outliers --------
new.data.modified<-subset(new.data,new.data$CreditScore>405 & Age<65)



#rf impute imputation
df2 <- new.data[,-c(3,6,5)] # removing three columns having char type
rf.data<-new.data 
set.seed(123)
new.rf.data<-rfImpute( Exited~.,data = df2)   # imputing empty values
rf.data$Balance<-new.rf.data$Balance         # adding imputed balance column to the data having all variables




#train test-------
set.seed(12345)
sample.size<-sample(2,nrow(rf.data),replace=TRUE , prob = c(0.7,0.3))
train<- rf.data[sample.size==1,]
test<- rf.data[sample.size==2,]

#checking class imbalance--
summary(train$Exited)



#correcting class imbalance ---------
over<-ovun.sample(Exited~.,data=train , method = 'over' , N=10862)$data
under<-ovun.sample(Exited~.,data=train , method = 'under' , N=2730)$data
both<-ovun.sample(Exited~.,data=train , method = 'both' , 
                  p=0.5,  seed=123,  N=5431)$data


#setting classifier

classifier.over=svm(formula=Exited~.,
                    data=over,
                    type='C-classification',
                    kernel = 'radial'
)

classifier.under=svm(formula=Exited~.,
                     data=under,
                     type='C-classification',
                     kernel = 'radial'
)
classifier.both=svm(formula=Exited~.,
                    data=both,
                    type='C-classification',
                    kernel = 'radial'
)

classifier=svm(formula=Exited~.,
               data=train,
               type='C-classification',
               kernel = 'radial'
)

#prediction using test 
pred=predict(classifier,newdata = train)
pred.over=predict(classifier.over,newdata = over)
pred.under=predict(classifier.under,newdata = under)
pred.both=predict(classifier.both,newdata = both)






#evaluation

cm=table(train$Exited,pred)
cm.over=table(over$Exited,pred.over)
cm.under=table(under$Exited,pred.under)
cm.both=table(both$Exited,pred.both)


prec1<- (cm[1,1])/sum(cm[1,1],cm[2,1])
rec1<- (cm[1,1])/sum(cm[1,1],cm[1,2])

prec2<- (cm.over[1,1])/sum(cm.over[1,1],cm.over[2,1])
rec2<- (cm.over[1,1])/sum(cm.over[1,1],cm.over[1,2])

prec3<- (cm.under[1,1])/sum(cm.under[1,1],cm.under[2,1])
rec3<- (cm.under[1,1])/sum(cm.under[1,1],cm.under[1,2])

prec4<- (cm.both[1,1])/sum(cm.both[1,1],cm.both[2,1])
rec4<- (cm.both[1,1])/sum(cm.both[1,1],cm.both[1,2])






confusionMatrix(train$Exited,pred)          #Accuracy : 
cat("\n precision",prec1)               #precision : 
cat("\n recall ",rec1)                  #recall : 
cat("\n f-score:",2*(prec1*rec1)/(prec1+rec1))   #f-score : 


confusionMatrix(over$Exited,pred.over)     #Accuracy : 
cat("\n precision",prec2)               #precision : 
cat("\n recall ",rec2)                  #recall : 
cat("\n f-score:",2*(prec2*rec2)/(prec2+rec2))   #f-score : 


confusionMatrix(under$Exited,pred.under)  #Accuracy : 
cat("\n precision",prec3)               #precision : 
cat("\n recall ",rec3)                  #recall : 
cat("\n f-score:",2*(prec3*rec1)/(prec3+rec3))   #f-score : 


confusionMatrix(both$Exited,pred.both)  # Accuracy :  
cat("\n precision",prec4)               #precision : 
cat("\n recall ",rec4)                  #recall : 
cat("\n f-score:",2*(prec4*rec4)/(prec4+rec4))   #f-score : 






#setting classifier - linear

classifier.1.over=svm(formula=Exited~.,
                      data=over,
                      type='C-classification',
                      kernel = 'linear'
)

classifier.1.under=svm(formula=Exited~.,
                       data=under,
                       type='C-classification',
                       kernel = 'linear'
)
classifier.1.both=svm(formula=Exited~.,
                      data=both,
                      type='C-classification',
                      kernel = 'linear'
)

classifier.1=svm(formula=Exited~.,
                 data=train,
                 type='C-classification',
                 kernel = 'linear'
)

#prediction using test 
pred.1=predict(classifier.1,newdata = train)
pred.1.over=predict(classifier.1.over,newdata = over)
pred.1.under=predict(classifier.1.under,newdata = under)
pred.1.both=predict(classifier.1.both,newdata = both)






#evaluation

cm.1=table(train$Exited,pred.1)
cm.1.over=table(over$Exited,pred.1.over)
cm.1.under=table(under$Exited,pred.1.under)
cm.1.both=table(both$Exited,pred.1.both)


prec1.1<- (cm.1[1,1])/sum(cm.1[1,1],cm.1[2,1])
rec1.1<- (cm.1[1,1])/sum(cm.1[1,1],cm.1[1,2])

prec2.1<- (cm.1.over[1,1])/sum(cm.1.over[1,1],cm.1.over[2,1])
rec2.1<- (cm.1.over[1,1])/sum(cm.1.over[1,1],cm.1.over[1,2])

prec3.1<- (cm.1.under[1,1])/sum(cm.1.under[1,1],cm.1.under[2,1])
rec3.1<- (cm.1.under[1,1])/sum(cm.1.under[1,1],cm.1.under[1,2])

prec4.1<- (cm.1.both[1,1])/sum(cm.1.both[1,1],cm.1.both[2,1])
rec4.1<- (cm.1.both[1,1])/sum(cm.1.both[1,1],cm.1.both[1,2])


confusionMatrix(train$Exited,pred.1)          #Accuracy :  
cat("\n precision",prec1.1)               #precision : 
cat("\n recall ",rec1.1)                  #recall : 
cat("\n f-score:",2*(prec1.1*rec1.1)/(prec1.1+rec1.1))   #f-score :




confusionMatrix(over$Exited,pred.1.over)     #Accuracy :     
cat("\n precision",prec2.1)               #precision : 
cat("\n recall ",rec2.1)                  #recall : 
cat("\n f-score:",2*(prec2.1*rec2.1)/(prec2.1+rec2.1))   #f-score : 



confusionMatrix(under$Exited,pred.1.under)  #Accuracy :     
cat("\n precision",prec3.1)               #precision :
cat("\n recall ",rec3.1)                  #recall : 
cat("\n f-score:",2*(prec3.1*rec3.1)/(prec3.1+rec3.1))   #f-score :



confusionMatrix(both$Exited,pred.1.both)  # Accuracy : 
cat("\n precision",prec4.1)               #precision : 
cat("\n recall ",rec4.1)                  #recall : 
cat("\n f-score:",2*(prec4.1*rec4.1)/(prec4.1+rec4.1))   #f-score : 



