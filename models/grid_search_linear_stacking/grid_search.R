K=3 #number of re-sampling (we use a different seed each time)
k=5 #number of folds used when training lvl 1 models (gbm and svr)

# we save the results of the search in "results"
results = list()
results["gbm_weight"]=numeric(0)
results["svr_weight"]=numeric(0)
results["rmsle"]=numeric(0)

for (s in 1:K){

    set.seed(s)
    folds=createFolds(train$SalePrice, k = k, list = TRUE)

    # fit and predict SalePrice on the whole train set, with k folds
    bootControl <- trainControl(number = 1, verboseIter=TRUE)
    tuneGrid = expand.grid(C=c(1.25),sigma=c(0.015)) # mandatory
    svr.train.predicted = data.table(SalePrice=numeric(nrow(train.kept_svr)))
    for (i in 1:k){ 
        train_i.sample = train.kept_svr[-folds[[i]],-"SalePrice",with=FALSE]
        train_i.target = train.kept_svr[-folds[[i]],.(SalePrice=as.numeric(SalePrice))]
        svrFit_i = train(x=train_i.sample,y=train_i.target$SalePrice,method='svmRadial',
                                   trControl=bootControl, tuneGrid=tuneGrid, preProcess=c("center","scale"))
        # predict remaining fold
        svr.train.predicted[folds[[i]],"SalePrice"] = predict(svrFit_i$finalModel,
                                  newdata=data.table(scale(train.kept_svr[folds[[i]],-"SalePrice",with=FALSE])))
    }

    bootControl <- trainControl(number = 1, verboseIter=TRUE)
    gbmGrid = expand.grid(interaction.depth = 4,n.trees = c(1950),shrinkage=c(.03),
                                n.minobsinnode=10)

    gbm.train.predicted = data.table(SalePrice=numeric(nrow(train.kept_gbm)))


    # fit and predict SalePrice on the whole train set, with k folds
    for (i in 1:k){ 
        print("seed: ")
        print(s)
        print("fold: ")
        print(i)
        train_i.sample = train.kept_gbm[-folds[[i]],-"SalePrice",with=FALSE]
        train_i.target = train.kept_gbm$SalePrice[-folds[[i]]]
        gbmFit_i <- train(train_i.sample,train_i.target,method='gbm',trControl=bootControl,verbose=TRUE,
                                 bag.fraction=.7,tuneGrid=gbmGrid,metric='RMSE')
        # predict remaining fold
        gbm.train.predicted[folds[[i]],"SalePrice"] = predict(gbmFit_i$finalModel,
                                  newdata=train.kept_gbm[folds[[i]],-"SalePrice",with=FALSE],n.trees=1950)
    }
    

    # create train set for lvl2 lm
    train_2.sample = data.table(preds_svr=svr.train.predicted$SalePrice,
                                preds_gbm=gbm.train.predicted$SalePrice,SalePrice=train$SalePrice)
    set.seed(K+s)
    train_a_part = createDataPartition(train$SalePrice,p=.80,list=FALSE)

    # fit lvl2 lm and get rmsle validation score
    lmFit = lm(SalePrice ~ 0 + preds_svr + preds_gbm, data=train_2.sample[train_a_part])
    lmFit.predict_b = predict(lmFit,newdata=train_2.sample[-train_a_part])
    
    # save results
    seed_rmsle = rmsle(train$SalePrice[-train_a_part],lmFit.predict_b)
    results[["rmsle"]] = c(results[["rmsle"]],seed_rmsle)
    results[["gbm_weight"]] = c(results[["gbm_weight"]],lmFit$coefficients[["preds_gbm"]])
    results[["svr_weight"]] = c(results[["svr_weight"]],lmFit$coefficients[["preds_svr"]])
    
    print(results)


}
