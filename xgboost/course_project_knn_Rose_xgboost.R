# packages
library('xgboost')
library('dplyr')
library('magrittr')
library('Matrix')


#reading csv file -------
data<-read.csv('Churn_Modelling.csv')

#converting from int to factor ----------
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


#KNN imputation-----
new.knn.data <- new.data[,-c(3,6,5)]
new.knn.imputed<-kNN(new.knn.data , variable= c('Balance') , k=7)
new.knn.imputed<-subset(new.knn.imputed , select = RowNumber : Exited)  


#train test-------
set.seed(12345)
sample.size<-sample(2,nrow(new.knn.imputed),replace=TRUE , prob = c(0.7,0.3))
train<- new.knn.imputed[sample.size==1,]
test<- new.knn.imputed[sample.size==2,]


#correcting class imbalance rose

#correcting class imbalance ---------
over<-ovun.sample(Exited~.,data=train , method = 'over' , N=11230)$data
under<-ovun.sample(Exited~.,data=train , method = 'under' , N=2786)$data
both<-ovun.sample(Exited~.,data=train , method = 'both' , 
                  p=0.5,  seed=123,  N=5615)$data


#create matrix ----------------------------------------------------------------

      #no sampling
trainm<- sparse.model.matrix(Exited~. , -11 , data=train)
train_label <-train[,"Exited"]
train_matrix<-xgb.DMatrix( data = as.matrix(trainm), label=train_label)

testm <- sparse.model.matrix(Exited~. , -11, data=test)
test_label<-test[,"Exited"]
test_matrix<-xgb.DMatrix(data=as.matrix(testm)  , label = test_label)




      #over
trainm.over<- sparse.model.matrix(Exited~. , -11 , data=over)
train_label.over <-over[,"Exited"]
train_matrix.over<-xgb.DMatrix( data = as.matrix(trainm.over), label=train_label.over)


testm.over <- sparse.model.matrix(Exited~. , -11, data=over)
test_label.over<-over[,"Exited"]
test_matrix.over<-xgb.DMatrix(data=as.matrix(testm.over)  , label = test_label.over)

    #under
trainm.under<- sparse.model.matrix(Exited~. , -11 , data=under)
train_label.under <-under[,"Exited"]
train_matrix.under<-xgb.DMatrix( data = as.matrix(trainm.under), label=train_label.under)


testm.under <- sparse.model.matrix(Exited~. , -11, data=under)
test_label.under<-under[,"Exited"]
test_matrix.under<-xgb.DMatrix(data=as.matrix(testm.under)  , label = test_label.under)

    #both
trainm.both<- sparse.model.matrix(Exited~. , -11 , data=both)
train_label.both <-both[,"Exited"]
train_matrix.both<-xgb.DMatrix( data = as.matrix(trainm.both), label=train_label.both)


testm.both <- sparse.model.matrix(Exited~. , -11, data=both)
test_label.both<-both[,"Exited"]
test_matrix.both<-xgb.DMatrix(data=as.matrix(testm.both)  , label = test_label.both)


#parameters--------------------------------

    #no sampling
nc<-length(unique(train_label))
xgb_params<-list("objective" = "multi:softprob",
                 "eval_metric" = "mlogloss" ,
                 "num_class"=nc)
watchlist<-list(train = train_matrix , test=test_matrix )

    #over
nc.over<-length(unique(train_label.over))
xgb_params.over<-list("objective" = "multi:softprob",
                 "eval_metric" = "mlogloss" ,
                 "num_class"=nc.over)
watchlist.over<-list(train = train_matrix.over , test=test_matrix.over )

    #under
nc.under<-length(unique(train_label.under))
xgb_params.under<-list("objective" = "multi:softprob",
                      "eval_metric" = "mlogloss" ,
                      "num_class"=nc.under)
watchlist.under<-list(train = train_matrix.under , test=test_matrix.under )


    #both
nc.both<-length(unique(train_label.both))
xgb_params.both<-list("objective" = "multi:softprob",
                       "eval_metric" = "mlogloss" ,
                       "num_class"=nc.both)
watchlist.both<-list(train = train_matrix.both , test=test_matrix.both )





#model--------------------------------------------
  
  #no sampling-
bst_model <-xgb.train(params = xgb_params ,
                      data=train_matrix,
                      nround = 100 ,
                      watchlist = watchlist,
                      eta = 0.05,
                      max.depth = 6,
                      gamma = 2)

    


    #over--
bst_model.over <-xgb.train(params = xgb_params.over ,
                      data=train_matrix.over,
                      nround = 100 ,
                      watchlist = watchlist.over,
                      eta = 0.05,
                      max.depth = 6,
                      gamma = 2)

   #under
bst_model.under <-xgb.train(params = xgb_params.under ,
                           data=train_matrix.under,
                           nround = 100 ,
                           watchlist = watchlist.under,
                           eta = 0.05,
                           max.depth = 6,
                           gamma = 2)

    #both
bst_model.both <-xgb.train(params = xgb_params.both ,
                            data=train_matrix.both,
                            nround = 100 ,
                            watchlist = watchlist.both,
                           eta = 0.05,
                           max.depth = 6,
                           gamma = 2)


#confusion matrix----------------------------------------------------------

    #no sampling-
p<-predict(bst_model , newdata = test_matrix)
pred<-matrix(p,nrow=nc,ncol=length(p)/nc) %>%
  t()%>%
  data.frame()%>%
  mutate(label=test_label , max_prob =max.col(.,"last")-1)

t<-table(prediction = pred$max_prob , Actual = pred$label)

acc<-sum(t[1,1],t[2,2])/sum(t[1,1],t[1,2],t[2,2],t[2,1])
sen<-t[1,1]/sum(t[1,1],t[2,1])
spe<-t[2,2]/sum(t[2,2],t[1,2])
pre<-t[1,1]/sum(t[1,1],t[1,2])
rec<-t[1,1]/sum(t[1,1],t[2,1])
f_s<- 2*pre*rec/sum(pre,rec);



    #over---
p.over<-predict(bst_model.over , newdata = test_matrix.over)
pred.over<-matrix(p.over,nrow=nc.over,ncol=length(p.over)/nc.over) %>%
  t()%>%
  data.frame()%>%
  mutate(label=test_label.over , max_prob =max.col(.,"last")-1)

t.over<-table(prediction = pred.over$max_prob , Actual = pred.over$label)
acc.over<-sum(t.over[1,1],t.over[2,2])/sum(t.over[1,1],t.over[1,2],t.over[2,2],t.over[2,1])
sen.over<-t.over[1,1]/sum(t.over[1,1],t.over[2,1])
spe.over<-t.over[2,2]/sum(t.over[2,2],t.over[1,2])
pre.over<-t.over[1,1]/sum(t.over[1,1],t.over[1,2])
rec.over<-t.over[1,1]/sum(t.over[1,1],t.over[2,1])
f_s.over<- 2*pre.over*rec.over/sum(pre.over,rec.over);




    #under
p.under<-predict(bst_model.under , newdata = test_matrix.under)
pred.under<-matrix(p.under,nrow=nc.under,ncol=length(p.under)/nc.under) %>%
  t()%>%
  data.frame()%>%
  mutate(label=test_label.under , max_prob =max.col(.,"last")-1)

t.under<-table(prediction = pred.under$max_prob , Actual = pred.under$label)
acc.under<-sum(t.under[1,1],t.under[2,2])/sum(t.under[1,1],t.under[1,2],t.under[2,2],t.under[2,1])
sen.under<-t.under[1,1]/sum(t.under[1,1],t.under[2,1])
spe.under<-t.under[2,2]/sum(t.under[2,2],t.under[1,2])
pre.under<-t.under[1,1]/sum(t.under[1,1],t.under[1,2])
rec.under<-t.under[1,1]/sum(t.under[1,1],t.under[2,1])
f_s.under<- 2*pre.under*rec.under/sum(pre.under,rec.under);

    #both
p.both<-predict(bst_model.both , newdata = test_matrix.both)
pred.both<-matrix(p.both,nrow=nc.both ,ncol=length(p.both)/nc.both) %>%
  t()%>%
  data.frame()%>%
  mutate(label=test_label.both , max_prob =max.col(.,"last")-1)

t.both<-table(prediction = pred.both$max_prob , Actual = pred.both$label)

acc.both<-sum(t.both[1,1],t.both[2,2])/sum(t.both[1,1],t.both[1,2],t.both[2,2],t.both[2,1])
sen.both<-t.both[1,1]/sum(t.both[1,1],t.both[2,1])
spe.both<-t.both[2,2]/sum(t.both[2,2],t.both[1,2])
pre.both<-t.both[1,1]/sum(t.both[1,1],t.both[1,2])
rec.both<-t.both[1,1]/sum(t.both[1,1],t.both[2,1])
f_s.both<- 2*pre.both*rec.both/sum(pre.both,rec.both);
    

#printing result
cat("accuracy : sensitivity : specificity : precision : recall : f_Score")
cat(acc,sen,spe,pre,rec,f_s)
cat("\n")
cat(acc.over,sen.over,spe.over,pre.over,rec.over,f_s.over)
cat("\n")
cat(acc.under,sen.under,spe.under,pre.under,rec.under,f_s.under)
cat("\n")
cat(acc.both,sen.both,spe.both,pre.both,rec.both,f_s.both)



#analysis
#plot
e<-data.frame(bst_model.over$evaluation_log)
plot(e$iter , e$train_mlogloss , col='blue')
lines(e$iter , e$test_mlogloss , col = 'red')
