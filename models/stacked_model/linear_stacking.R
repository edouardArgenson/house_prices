
library('lattice')
library('splines')
library('parallel')
library('survival')
library('ggplot2')
library('caret')
library('data.table')
library('Metrics')
library('MASS')
library('e1071')
library('kernlab')
library('gbm')
library('plyr')

train = fread('~/kaggle/house_prices/data/train.csv',
              colClasses=c('MiscFeature'='character','PoolQC'='character','Alley'='character'))

# Rename columns 1stFlrSF, 2ndFlrSF, and 3SsnPorch
FirstFlrSF=train$'1stFlrSF'
SecondFlrSF=train$'2ndFlrSF'
ThreeSsnPorch=train$'3SsnPorch'
new_names = names(train)[-which(names(train)=='1stFlrSF'|names(train)=='2ndFlrSF'|names(train)=='3SsnPorch')]
to_add = data.table(FirstFlrSF,SecondFlrSF,ThreeSsnPorch)
train = cbind(train[,new_names,with=FALSE],to_add)

# Transform categorical arguments KitchenQual, ExterQual, BsmtQual, GarageFinish, into numerical

# KitchenQual
nKitchenQual = numeric(length(train$KitchenQual))
nKitchenQual[train$KitchenQual=='TA']=1.0
nKitchenQual[train$KitchenQual=='Gd']=2.0
nKitchenQual[train$KitchenQual=='Ex']=3.0
train=cbind(train,nKitchenQual)

# ExterQual
nExterQual = numeric(length(train$ExterQual))
nExterQual[train$ExterQual=='TA']=1.0
nExterQual[train$ExterQual=='Gd']=2.0
nExterQual[train$ExterQual=='Ex']=3.0
train=cbind(train,nExterQual)

# BsmtQual
nBsmtQual = numeric(length(train$BsmtQual))
nBsmtQual[train$BsmtQual=='TA']=1.0
nBsmtQual[train$BsmtQual=='Gd']=2.0
nBsmtQual[train$BsmtQual=='Ex']=3.0
train=cbind(train,nBsmtQual)

# GarageFinish
nGarageFinish = numeric(length(train$GarageFinish))
nGarageFinish[train$GarageFinish=='Unf']=1.0
nGarageFinish[train$GarageFinish=='RFn']=2.0
nGarageFinish[train$GarageFinish=='Fin']=3.0
train=cbind(train,nGarageFinish)

# Full and half bathrooms
train$Bath = train$FullBath + train$HalfBath
train$BsmtBaths = train$BsmtFullBath + train$BsmtHalfBath

# TotalBsmtSF_on_GRLivArea (for SVR)
train$TotalBsmtSF_on_GrLivArea = train$TotalBsmtSF/train$GrLivArea

# MSSubClassCat
train$MSSubClassCat = train[,.(MSSubClassCat=sapply(MSSubClass,toString)),with=TRUE]

# load test file

test = fread('~/kaggle/house_prices/data/test.csv',
              colClasses=c('MiscFeature'='character','PoolQC'='character','Alley'='character'))

# Il faut renommer les colonnes 1stFlrSF, 2ndFlrSF, et 3SsnPorch pour pas avoir d'emmerdes
FirstFlrSF=test$'1stFlrSF'
SecondFlrSF=test$'2ndFlrSF'
ThreeSsnPorch=test$'3SsnPorch'
new_names = names(test)[-which(names(test)=='1stFlrSF'|names(test)=='2ndFlrSF'|names(test)=='3SsnPorch')]
to_add = data.table(FirstFlrSF,SecondFlrSF,ThreeSsnPorch)
test = cbind(test[,new_names,with=FALSE],to_add)

# Transform categorical arguments KitchenQual, ExterQual, BsmtQual, GarageFinish, into numerical

# KitchenQual
nKitchenQual = numeric(length(test$KitchenQual))
nKitchenQual[test$KitchenQual=='TA']=1.0
nKitchenQual[test$KitchenQual=='Gd']=2.0
nKitchenQual[test$KitchenQual=='Ex']=3.0
test=cbind(test,nKitchenQual)

# ExterQual
nExterQual = numeric(length(test$ExterQual))
nExterQual[test$ExterQual=='TA']=1.0
nExterQual[test$ExterQual=='Gd']=2.0
nExterQual[test$ExterQual=='Ex']=3.0
test=cbind(test,nExterQual)

# BsmtQual
nBsmtQual = numeric(length(test$BsmtQual))
nBsmtQual[test$BsmtQual=='TA']=1.0
nBsmtQual[test$BsmtQual=='Gd']=2.0
nBsmtQual[test$BsmtQual=='Ex']=3.0
test=cbind(test,nBsmtQual)

# GarageFinish
nGarageFinish = numeric(length(test$GarageFinish))
nGarageFinish[test$GarageFinish=='Unf']=1.0
nGarageFinish[test$GarageFinish=='RFn']=2.0
nGarageFinish[test$GarageFinish=='Fin']=3.0
test=cbind(test,nGarageFinish)

# Full and half bathrooms
test$Bath = test$FullBath + test$HalfBath
test$BsmtBaths = test$BsmtFullBath + test$BsmtHalfBath

# TotalBsmtSF_on_GrLivArea
test$TotalBsmtSF_on_GrLivArea = test$TotalBsmtSF/test$GrLivArea

# MSSubClassCat
test$MSSubClassCat = test[,.(MSSubClassCat=sapply(MSSubClass,toString)),with=TRUE]


# Deal with missing values
LotFrontage_mean = round(mean(train$LotFrontage,na.rm=TRUE))
train[which(is.na(LotFrontage)),'LotFrontage'] <- LotFrontage_mean
train=cbind(train,"IsGarage"=1+numeric(nrow(train)))
train[which(is.na(GarageYrBlt)),'GarageYrBlt'] <- 1900
#train[which(is.na(GarageQual)),'IsGarage'] <- 0
train[which(is.na(MasVnrArea)),'MasVnrArea'] <- 0
train[which(is.na(BsmtCond)),'BsmtCond'] <- 'MISSING'
train[which(is.na(BsmtFinType1)),'BsmtFinType1'] <- 'MISSING'
train[which(is.na(BsmtFinType2)),'BsmtFinType2'] <- 'MISSING'
train[which(is.na(BsmtFinSF1)),'BsmtFinSF1'] <- 0
train[which(is.na(BsmtFinSF2)),'BsmtFinSF2'] <- 0
train[which(is.na(TotalBsmtSF)),'TotalBsmtSF'] <- 0
train[which(is.na(GarageCars)),'GarageCars'] <- 0
train[which(is.na(GarageArea)),'GarageArea'] <- 0
train[which(is.na(BsmtUnfSF)),'BsmtUnfSF'] <- 0
train[which(is.na(BsmtFullBath)),'BsmtFullBath'] <- 0
train[which(is.na(BsmtHalfBath)),'BsmtHalfBath'] <- 0
train[which(is.na(MSZoning)),'MSZoning'] <- 'RL'
train[which(is.na(SaleType)),'SaleType'] <- 'Oth'
train[which(is.na(Exterior1st)),'Exterior1st'] <- 'Other'
train[which(is.na(Exterior2nd)),'Exterior2nd'] <- 'Other'
train[which(is.na(Functional)),'Functional'] <- 'Typ'

# create svr train set
kept_features_svr = c("LotArea","OverallQual","YearBuilt","YearRemodAdd","nKitchenQual","nExterQual",
                    "nBsmtQual","GrLivArea","Bath","nGarageFinish",
                    "BsmtFinSF1","GarageCars","TotalBsmtSF","KitchenAbvGr","BedroomAbvGr","TotRmsAbvGrd",
                    "OverallCond","TotalBsmtSF_on_GrLivArea")
train.kept_svr = train[,c(kept_features_svr,"SalePrice"),with=FALSE]


# create gbm train set
kept_num_features_gbm = c("LotFrontage", "LotArea", "OverallQual", "OverallCond",  
                        "YearBuilt", "YearRemodAdd", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                        "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath",
                        "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
                        "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
                        "EnclosedPorch", "ThreeSsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold",
                        "YrSold","SalePrice")
kept_cat_features_gbm = c("Neighborhood","ExterQual","HeatingQC","CentralAir","KitchenQual","SaleType",
                  "SaleCondition","IsGarage")
kept_features_gbm = c(kept_num_features_gbm,kept_cat_features_gbm)
# Separate numeric and categorical features for conversion (as numeric and factor)
train_gbm.sample.num_features = train[,kept_num_features_gbm,with=FALSE]
train_gbm.sample.cat_features = train[,kept_cat_features_gbm,with=FALSE]
# Change class of data and merge back numeric and categorical
train_gbm.sample.num_features.toFit = train_gbm.sample.num_features[,lapply(.SD,as.numeric)]
train_gbm.sample.cat_features.toFit = train_gbm.sample.cat_features[,lapply(.SD,as.factor)]
train.kept_gbm = cbind(train_gbm.sample.num_features.toFit,train_gbm.sample.cat_features.toFit)



# Deal with missing values
LotFrontage_mean = round(mean(test$LotFrontage,na.rm=TRUE))
test[which(is.na(LotFrontage)),'LotFrontage'] <- LotFrontage_mean
test=cbind(test,"IsGarage"=1+numeric(nrow(test)))
test[which(is.na(GarageYrBlt)),'GarageYrBlt'] <- 1900
#test[which(is.na(GarageQual)),'IsGarage'] <- 0
test[which(is.na(MasVnrArea)),'MasVnrArea'] <- 0
test[which(is.na(BsmtCond)),'BsmtCond'] <- 'MISSING'
test[which(is.na(BsmtFinType1)),'BsmtFinType1'] <- 'MISSING'
test[which(is.na(BsmtFinType2)),'BsmtFinType2'] <- 'MISSING'
test[which(is.na(BsmtFinSF1)),'BsmtFinSF1'] <- 0
test[which(is.na(BsmtFinSF2)),'BsmtFinSF2'] <- 0
test[which(is.na(TotalBsmtSF)),'TotalBsmtSF'] <- 0
test[which(is.na(GarageCars)),'GarageCars'] <- 0
test[which(is.na(GarageArea)),'GarageArea'] <- 0
test[which(is.na(BsmtUnfSF)),'BsmtUnfSF'] <- 0
test[which(is.na(BsmtFullBath)),'BsmtFullBath'] <- 0
test[which(is.na(BsmtHalfBath)),'BsmtHalfBath'] <- 0
test[which(is.na(MSZoning)),'MSZoning'] <- 'RL'
test[which(is.na(SaleType)),'SaleType'] <- 'Oth'
test[which(is.na(Exterior1st)),'Exterior1st'] <- 'Other'
test[which(is.na(Exterior2nd)),'Exterior2nd'] <- 'Other'
test[which(is.na(Functional)),'Functional'] <- 'Typ'
test$BsmtBaths = test$BsmtFullBath + test$BsmtHalfBath
test$TotalBsmtSF_on_GrLivArea = test$TotalBsmtSF/test$GrLivArea

# create svr test set
kept_features_svr = c("LotArea","OverallQual","YearBuilt","YearRemodAdd","nKitchenQual","nExterQual",
                   "nBsmtQual","GrLivArea","Bath","nGarageFinish",
                   "BsmtFinSF1","GarageCars","TotalBsmtSF","KitchenAbvGr","BedroomAbvGr","TotRmsAbvGrd","OverallCond",
                   "TotalBsmtSF_on_GrLivArea")
test.kept_svr = test[,c(kept_features_svr),with=FALSE]


# create gbm test set
kept_num_features_gbm = c("LotFrontage", "LotArea", "OverallQual", "OverallCond",  
                        "YearBuilt", "YearRemodAdd", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                        "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath",
                        "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
                        "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
                        "EnclosedPorch", "ThreeSsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold")
kept_cat_features_gbm = c("Neighborhood","ExterQual","HeatingQC","CentralAir","KitchenQual","SaleType",
                  "SaleCondition","IsGarage")
kept_features_gbm = c(kept_num_features_gbm,kept_cat_features_gbm)
# Separate numeric and categorical features for conversion (as numeric and factor)
test_gbm.sample.num_features = test[,kept_num_features_gbm,with=FALSE]
test_gbm.sample.cat_features = test[,kept_cat_features_gbm,with=FALSE]
# Change class of data and merge back numeric and categorical
test_gbm.sample.num_features.toPredict = test_gbm.sample.num_features[,lapply(.SD,as.numeric)]
test_gbm.sample.cat_features.toPredict = test_gbm.sample.cat_features[,lapply(.SD,as.factor)]
test.kept_gbm = cbind(test_gbm.sample.num_features.toPredict,test_gbm.sample.cat_features.toPredict)



# separate train set in k fols
k=5

set.seed(13)
folds=createFolds(train$SalePrice, k = k, list = TRUE)


list_rmsle = list()

# for each fold i, do: {fit a SVR on the k-1 other folds then predict values for the fold i}

bootControl <- trainControl(number = 1, verboseIter=TRUE)
tuneGrid = expand.grid(C=c(1.25),sigma=c(0.015)) # mandatory

svr.train.predicted = data.table(SalePrice=numeric(nrow(train.kept_svr)))

for (i in 1:k){ 
    # Remarque: en fait on n'a même pas besoin de stocker les modèles entraînés, car seules les 
    # prédictions nous intéressent.
    
    #train_fold_i = Reduce(c,folds[-i])
    #to_predict_fold_i = folds[[i]]
    train_i.sample = train.kept_svr[-folds[[i]],-"SalePrice",with=FALSE]
    train_i.target = train.kept_svr[-folds[[i]],.(SalePrice=as.numeric(SalePrice))]

    svrFit_i = train(x=train_i.sample,y=train_i.target$SalePrice,method='svmRadial',
                          trControl=bootControl, tuneGrid=tuneGrid, preProcess=c("center","scale"))
    
    # predict remaining fold
    
    svr.train.predicted[folds[[i]],"SalePrice"] = predict(svrFit_i$finalModel,
                            newdata=data.table(scale(train.kept_svr[folds[[i]],-"SalePrice",with=FALSE])))
    
    
}

list_rmsle["lvl1_svr"] = rmsle(svr.train.predicted$SalePrice,train$SalePrice)


# for each fold i, do: {fit a gbm model on the k-1 other folds then predict values for the fold i}

bootControl <- trainControl(number = 1, verboseIter=TRUE)
gbmGrid = expand.grid(interaction.depth = 4,n.trees = c(1950),shrinkage=c(.03),
                      n.minobsinnode=10)

gbm.train.predicted = data.table(SalePrice=numeric(nrow(train.kept_gbm)))

#gbmFit_i <- 0
#gc(verbose=TRUE)

for (i in 1:k){ 
  
    train_i.sample = train.kept_gbm[-folds[[i]],-"SalePrice",with=FALSE]
    train_i.target = train.kept_gbm$SalePrice[-folds[[i]]]

    gbmFit_i <- train(train_i.sample,train_i.target,method='gbm',trControl=bootControl,verbose=TRUE,
               bag.fraction=.7,tuneGrid=gbmGrid,metric='RMSE')
    
    # predict remaining fold
    gbm.train.predicted[folds[[i]],"SalePrice"] = predict(gbmFit_i$finalModel,
                            newdata=train.kept_gbm[folds[[i]],-"SalePrice",with=FALSE],n.trees=1950)
    
    #gbmFit_i <- 0
    #gc(verbose=TRUE) 
}

list_rmsle["lvl1_gbm"] = rmsle(gbm.train.predicted$SalePrice,train$SalePrice)


# create train set for lvl2 lm

train_2.sample = data.table(preds_svr=svr.train.predicted$SalePrice,
                            preds_gbm=gbm.train.predicted$SalePrice,SalePrice=train$SalePrice)

set.seed(10)
train_a_part = createDataPartition(train$SalePrice,p=.80,list=FALSE)

head(train_2.sample)

# fit lvl2 lm

# grid search result: n.trees = 205, interaction.depth = 1, shrinkage = 0.05, n.minobsinnode = 12
# for the following search:
#gbmGrid <- expand.grid(interaction.depth = (1:3),n.trees = (30:50)*5, 
#                       shrinkage = c(.02,.03,.04,.05,.06,.07,.08),n.minobsinnode = (5:20))

# ----------------------------------
## checking rmsle score on a fold

#bootControl <- trainControl(number = 10, verboseIter=TRUE)
#lmFit = train(SalePrice ~ preds_svr+preds_gbm-1,method='lm',data=train_2.sample,
#              trControl=bootControl,metric="RMSE")
#lmFit.predict_b = predict(lmFit$finalModel,newdata=train_2.sample[-train_a_part])
#
#list_rmsle["lvl2_gbm"] = rmsle(train$SalePrice[-train_a_part],lmFit.predict_b)

# -----
# OR
# ----- 

# ----------------------------------
# training on full train set for predictions

#bootControl <- trainControl(number = 10, verboseIter=TRUE)
#lmFit = train(SalePrice ~ 0 + preds_svr + preds_gbm, method='lm',data=train_2.sample,
#              trControl=bootControl,metric="RMSE")

lmFit = lm(SalePrice ~ 0 + preds_svr + preds_gbm, data=train_2.sample)

# fit lvl1 models on full train set, and predict test set SalePrices


# -----------------------
# SVR

# train
bootControl <- trainControl(number = 1, verboseIter=TRUE)
tuneGrid = expand.grid(C=c(1.25),sigma=c(0.015)) # mandatory

train_svr.sample = train.kept_svr[,-"SalePrice",with=FALSE]
train_svr.target = train.kept_svr[,.(SalePrice=as.numeric(SalePrice))]

svrFit = train(x=train_svr.sample,y=train_svr.target$SalePrice,method='svmRadial',
                          trControl=bootControl, tuneGrid=tuneGrid, preProcess=c("center","scale"))
    
# predict
svr.test.preds = predict(svrFit$finalModel,
                        newdata=data.table(scale(test.kept_svr)))
    
    
# -----------------------
# gbm

# train
bootControl <- trainControl(number = 1, verboseIter=TRUE)
gbmGrid = expand.grid(interaction.depth=4,n.trees=c(1950),shrinkage=c(.03),n.minobsinnode=10)
  
train_gbm.sample = train.kept_gbm[,-"SalePrice",with=FALSE]
train_gbm.target = train.kept_gbm$SalePrice

gbmFit <- train(train_gbm.sample,train_gbm.target,method='gbm',trControl=bootControl,verbose=TRUE,
               bag.fraction=.7,tuneGrid=gbmGrid,metric='RMSE')
    
# predict
gbm.test.preds = predict(gbmFit$finalModel,
                            newdata=test.kept_gbm,n.trees=1950)


# -----------------------
# create train set for lvl2 gbm
test_2.sample = data.table(preds_svr=svr.test.preds,preds_gbm=gbm.test.preds)
setnames(test_2.sample,c("preds_svr","preds_gbm"))

# predict with lvl2 model and write submission file

lmFit.test.preds = predict(lmFit,newdata=test_2.sample)

test.sample_submission = fread('~/kaggle/house_prices/data/sample_submission.csv')
test.sample_submission = test.sample_submission[,.(Id)]
test.sample_submission.new = cbind(test.sample_submission,SalePrice=lmFit.test.preds)

setnames(test.sample_submission.new,c("Id","SalePrice"))
write.csv(test.sample_submission.new,'~/kaggle/house_prices/data/linear_stacked_with_cst.csv',row.names=FALSE)

# leaderboard score 0.12677 with cte
# leaderboard score without cte: ??

# check lvl1 gbm leaderboard score

#test.sample_submission = fread('~/kaggle/house_prices/data/sample_submission.csv')
#test.sample_submission = test.sample_submission[,.(Id)]
#test.sample_submission.new = cbind(test.sample_submission,SalePrice=gbm.test.preds)
#write.csv(test.sample_submission.new,'~/kaggle/house_prices/data/gbm_test_submit.csv',row.names=FALSE)

# leaderboard score 0.12827

# check lvl1 svr leaderboard score

#test.sample_submission = fread('~/kaggle/house_prices/data/sample_submission.csv')
#test.sample_submission = test.sample_submission[,.(Id)]
#test.sample_submission.new = cbind(test.sample_submission,SalePrice=svr.test.preds)
#write.csv(test.sample_submission.new,'~/kaggle/house_prices/data/svr_test_submit.csv',row.names=FALSE)

# leaderboard score 0.13999,
