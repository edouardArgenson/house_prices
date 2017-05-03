
library('lattice')
library('ggplot2')
library('caret')
library('data.table')
library('Metrics')
library('MASS')
library('e1071')
library('kernlab')
library('gbm')
library('survival')
library('splines')
library('parallel')
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


## Deal with missing values
#test[which(is.na(LotFrontage)),'LotFrontage'] <- LotFrontage_mean
#test=cbind(test,"IsGarage"=1+numeric(nrow(test)))
#test[which(is.na(GarageYrBlt)),'GarageYrBlt'] <- 1900
##test.sample[which(is.na(GarageQual)),'IsGarage'] <- 0
#test[which(is.na(MasVnrArea)),'MasVnrArea'] <- 0
#test[which(is.na(BsmtCond)),'BsmtCond'] <- 'MISSING'
#test[which(is.na(BsmtFinType1)),'BsmtFinType1'] <- 'MISSING'
#test[which(is.na(BsmtFinType2)),'BsmtFinType2'] <- 'MISSING'
#test[which(is.na(BsmtFinSF1)),'BsmtFinSF1'] <- 0
#test[which(is.na(BsmtFinSF2)),'BsmtFinSF2'] <- 0
#test[which(is.na(TotalBsmtSF)),'TotalBsmtSF'] <- 0
#test[which(is.na(GarageCars)),'GarageCars'] <- 0
#test[which(is.na(GarageArea)),'GarageArea'] <- 0
#test[which(is.na(BsmtUnfSF)),'BsmtUnfSF'] <- 0
#test[which(is.na(BsmtFullBath)),'BsmtFullBath'] <- 0
#test[which(is.na(BsmtHalfBath)),'BsmtHalfBath'] <- 0
#test[which(is.na(MSZoning)),'MSZoning'] <- 'RL'
#test[which(is.na(SaleType)),'SaleType'] <- 'Oth'
#test[which(is.na(Exterior1st)),'Exterior1st'] <- 'Other'
#test[which(is.na(Exterior2nd)),'Exterior2nd'] <- 'Other'
#test[which(is.na(Functional)),'Functional'] <- 'Typ'
#test$BsmtBaths = test$BsmtFullBath + test$BsmtHalfBath

#train.kept = train[,kept_features,with=FALSE]
#test.kept = test[,kept_features[-which(kept_features=="SalePrice")],with=FALSE]

# separate train set in two parts: train_a and train_b
# train_a for fitting base models
# train_b for fitting stage 2 model

set.seed(10)
train_a_part = createDataPartition(train$SalePrice,p=.80,list=FALSE)

#train.sample = train.kept[inTrain,-"SalePrice",with=FALSE]
#train.target = train.kept[inTrain,.(SalePrice=as.numeric(SalePrice))]
#test.sample = train.kept[-inTrain,-"SalePrice",with=FALSE]
#test.target = train.kept[-inTrain,.(SalePrice=as.numeric(SalePrice))]



# fit SVR model on train_a
# meta params: C=1.25, sigma=0.015 

kept_features_svr = c("LotArea","OverallQual","YearBuilt","YearRemodAdd","nKitchenQual","nExterQual",
                   "nBsmtQual","GrLivArea","Bath","nGarageFinish",
                   "BsmtFinSF1","GarageCars","TotalBsmtSF","KitchenAbvGr","BedroomAbvGr","TotRmsAbvGrd","OverallCond",
                   "TotalBsmtSF_on_GrLivArea")

train.kept_svr = train[,c(kept_features_svr,"SalePrice"),with=FALSE]

train_a.sample = train.kept_svr[train_a_part,-"SalePrice",with=FALSE]
train_a.target = train.kept_svr[train_a_part,.(SalePrice=as.numeric(SalePrice))]


bootControl <- trainControl(number = 10, verboseIter=TRUE)
tuneGrid = expand.grid(C=c(1.25),sigma=c(0.015)) # mandatory

svrFit_a = train(x=train_a.sample,y=train_a.target$SalePrice,method='svmRadial',trControl=bootControl,
               tuneGrid=tuneGrid, preProcess=c("center","scale"))



# predict train_b with SVR model

train_b.sample = data.table(scale(train.kept_svr[-train_a_part,-"SalePrice",with=FALSE]))
train_b.target = train.kept_svr[-train_a_part,.(SalePrice=as.numeric(SalePrice))]

svrFit_a.predict_b = predict(svrFit_a$finalModel,newdata=train_b.sample)

print("train_b.sample SalePrice predicted with model svrFit_a")

# print rmsle
print("rmsle:")
print(rmsle(train_b.target$SalePrice,svrFit_a.predict_b))

# fit gbm model on train_a
# meta parameters: 1950 trees, depth=4, shrinkage=.03


kept_num_features_gbm = c("LotFrontage", "LotArea", "OverallQual", "OverallCond",  
                        "YearBuilt", "YearRemodAdd", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                        "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath",
                        "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
                        "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
                        "EnclosedPorch", "ThreeSsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold")
kept_cat_features_gbm = c("Neighborhood","ExterQual","HeatingQC","CentralAir","KitchenQual","SaleType",
                  "SaleCondition","IsGarage")
kept_features_gbm = c(kept_num_features_gbm,kept_cat_features_gbm)


train.kept_gbm = train[,c(kept_features_gbm,"SalePrice"),with=FALSE]

# Separate numeric and categorical features for conversion (as numeric and factor)
train_a.sample.num_features = train[train_a_part,kept_num_features_gbm,with=FALSE]
train_a.sample.cat_features = train[train_a_part,kept_cat_features_gbm,with=FALSE]
# Change class of data and merge back numeric and categorical
train_a.sample.num_features.toFit = train_a.sample.num_features[,lapply(.SD,as.numeric)]
train_a.sample.cat_features.toFit = train_a.sample.cat_features[,lapply(.SD,as.factor)]
train_a.sample = cbind(train_a.sample.num_features.toFit,train_a.sample.cat_features.toFit)

train_a.target = train.kept_gbm[train_a_part,.(SalePrice=as.numeric(SalePrice))]

bootControl <- trainControl(number = 10, verboseIter=TRUE)
gbmGrid = expand.grid(interaction.depth = (3:5),n.trees = c(1950),shrinkage=c(.02,.03,.04),
                      n.minobsinnode=10)

gbmFit_a = train(train_a.sample,train_a.target$SalePrice,method='gbm',trControl=bootControl,verbose=TRUE,
               bag.fraction=.8,tuneGrid=gbmGrid,metric='RMSE')

# .1353

# predict train_b with gbm model


# Separate numeric and categorical features for conversion (as numeric and factor)
train_b.sample.num_features = train[-train_a_part,kept_num_features_gbm,with=FALSE]
train_b.sample.cat_features = train[-train_a_part,kept_cat_features_gbm,with=FALSE]
# Change class of data and merge back numeric and categorical
train_b.sample.num_features.toFit = train_b.sample.num_features[,lapply(.SD,as.numeric)]
train_b.sample.cat_features.toFit = train_b.sample.cat_features[,lapply(.SD,as.factor)]
train_b.sample = cbind(train_b.sample.num_features.toFit,train_b.sample.cat_features.toFit)

train_b.target = train.kept_gbm[-train_a_part,.(SalePrice=as.numeric(SalePrice))]



gbmFit_a.predict_b = predict(gbmFit_a$finalModel,newdata=train_b.sample,n.trees=1950)

print("train_b.sample SalePrice predicted with model gbmFit_a")

# print rmsle
print("rmsle:")
print(rmsle(train_b.target$SalePrice,gbmFit_a.predict_b))

# Create new data.table with predictions on train_b, for level 1 model training

#train_2 = data.table(preds_svr=svrFit_a.predict_b,
#                     preds_gbm=gbmFit_a.predict_b,SalePrice=train[-train_a_part,SalePrice])

train_2.sample = data.table(preds_svr=svrFit_a.predict_b,preds_gbm=gbmFit_a.predict_b)
train_2.target = data.table(SalePrice=train[-train_a_part,SalePrice])


#head(train_2.sample)
#head(train_2.target)

# Fitting a gbm as level 1 model

gbmGrid <- expand.grid(interaction.depth = (1:3),n.trees = (30:40)*5, 
                       shrinkage = c(.02,.03,.04,.05,.06,.07,.08),n.minobsinnode = (2:10))
bootControl <- trainControl(number = 10, verboseIter=TRUE)


gbmFit_2 = train(train_2.sample,train_2.target$SalePrice,method='gbm',trControl=bootControl,verbose=TRUE,
               bag.fraction=.6,tuneGrid=gbmGrid,metric='RMSE')


# grid-search result:
# n.trees = 165, interaction.depth = 1, shrinkage = 0.05, n.minobsinnode = 5

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
test[which(is.na(LotFrontage)),'LotFrontage'] <- LotFrontage_mean
test=cbind(test,"IsGarage"=1+numeric(nrow(test)))
test[which(is.na(GarageYrBlt)),'GarageYrBlt'] <- 1900
#test.sample[which(is.na(GarageQual)),'IsGarage'] <- 0
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

#train.kept = train[,kept_features,with=FALSE]
#test.kept = test[,kept_features[-which(kept_features=="SalePrice")],with=FALSE]

# predict test with lvl0 SVR and gbm models

#---------------------------------
# SVR

test.sample_svr = data.table(scale(test[,kept_features_svr,with=FALSE])) # don't forget to scale
svrFit_a.test_preds = predict(svrFit_a$finalModel,newdata=test.sample_svr)


#---------------------------------
# gbm


# Separate numeric and categorical features for conversion (as numeric and factor)
test.sample.num_features_gbm = test[,kept_num_features_gbm,with=FALSE]
test.sample.cat_features_gbm = test[,kept_cat_features_gbm,with=FALSE]
# Change class of data and merge back numeric and categorical
test.sample.num_features_gbm.tp = test.sample.num_features_gbm[,lapply(.SD,as.numeric)]
test.sample.cat_features_gbm.tp = test.sample.cat_features_gbm[,lapply(.SD,as.factor)]
test.sample_gbm = cbind(test.sample.num_features_gbm.tp,test.sample.cat_features_gbm.tp)

gbmFit_a.test_preds = predict(gbmFit_a$finalModel,newdata=test.sample_gbm,n.trees=1950)

# build lvl1 test set

test_2.sample = data.table(test_preds_svr=svrFit_a.test_preds,test_preds_gbm=gbmFit_a.test_preds)

# predict test with lvl1 gbm aka gbmFit_2
gbmFit_2.test_preds = predict(gbmFit_2$finalModel,newdata=test_2.sample,n.trees=165)




# write submission file

test.sample_submission = fread('~/kaggle/house_prices/data/sample_submission.csv')
test.sample_submission = test.sample_submission[,.(Id)]
test.sample_submission.new = cbind(test.sample_submission,SalePrice=gbmFit_2.test_preds)
write.csv(test.sample_submission.new,'~/kaggle/house_prices/data/my_submission_stacked.csv',row.names=FALSE)

# leaderboard score = 0.13956 (with train_a=60% of dataset)


