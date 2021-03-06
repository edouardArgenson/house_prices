
library('lattice')
library('ggplot2')
library('caret')
library('data.table')
library('Metrics')
library('MASS')
library('e1071')

train = fread('~/kaggle/house_prices/data/train.csv',
              colClasses=c('MiscFeature'='character','PoolQC'='character'))

# Il faut renommer les colonnes 1stFlrSF, 2ndFlrSF, et 3SsnPorch pour pas avoir d'emmerdes
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

# Take log of SalePrice
train$log_SalePrice = train[,.(log_SalePrice=log(SalePrice))]

# Take log of a few features
train$log_LotArea = train[,.(log_LotArea=log(LotArea))]
train$log_OverallQual = train[,.(log_OverallQual=log(OverallQual))]
train$log_YearBuilt = train[,.(log_YearBuilt=log(YearBuilt))]
train$log_YearRemodAdd = train[,.(log_YearRemodAdd=log(YearRemodAdd))]
train$log_GrLivArea = train[,.(log_GrLivArea=log(GrLivArea))]
train$log_Bath = train[,.(log_Bath=log(1+Bath))]
train$log_nKitchenQual = train[,.(log_nKitchenQual=log(1+nKitchenQual))]
train$log_nBsmtQual = train[,.(log_nBsmtQual=log(1+nBsmtQual))]
train$log_nExterQual = train[,.(log_nExterQual=log(1+nExterQual))]
train$log_nGarageFinish = train[,.(log_nGarageFinish=log(1+nGarageFinish))]
train$log_BsmtFinSF1 = train[,.(log(1+BsmtFinSF1))]
train$log_GarageCars = train[,.(log(1+GarageCars))]
train$log_TotalBsmtSF = train[,.(log(1+TotalBsmtSF))]
train$log_KitchenAbvGr = train[,.(log(1+KitchenAbvGr))]
train$log_BedroomAbvGr = train[,.(log(1+BedroomAbvGr))]
train$log_TotRmsAbvGrd = train[,.(log(TotRmsAbvGrd))]
train$log_OverallCond = train[,.(log(OverallCond))]

# Try exp of a few features
# None useful

# Try powers of a few features
train$OverallQual_Square = train$OverallQual*train$OverallQual
train$OverallQual_3 = train$OverallQual*train$OverallQual*train$OverallQual
train$GrLivArea_Square = train$GrLivArea*train$GrLivArea
train$TotalBsmtSF_on_GrLivArea = train$TotalBsmtSF/train$GrLivArea
train$OverallCond_sqrt = sqrt(train$OverallCond)
train$OverallCond_square = train$OverallCond*train$OverallCond
train$LotArea_sqrt = sqrt(train$LotArea)
train$FirstFlrSF_sqrt = sqrt(train$FirstFlrSF)
train$TotRmsAbvGrd_sqrt = sqrt(train$TotRmsAbvGrd)

kept_features8 = c("LotArea","OverallQual","YearBuilt","YearRemodAdd","nKitchenQual","nExterQual",
                   "nBsmtQual","GrLivArea","Bath","nGarageFinish",
                   "BsmtFinSF1","GarageCars","TotalBsmtSF","KitchenAbvGr","BedroomAbvGr","TotRmsAbvGrd","OverallCond",
                   "TotalBsmtSF_on_GrLivArea")


        #"LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "BsmtFinSF1", "BsmtFinSF2",
        #         "BsmtUnfSF", "TotalBsmtSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath",
        #        "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
        #       "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch", 
        #      "MiscVal", "MoSold", "YrSold","PoolArea","FirstFlrSF","SecondFlrSF","ThreeSsnPorch","nKitchenQual",
        #     "nExterQual","nBsmtQual","nGarageFinish"
                
train.kept8 = train[,c(kept_features8,"SalePrice"),with=FALSE]

set.seed(5)
inTrain = createDataPartition(train.kept8$SalePrice,p=.75,list=FALSE)
#print(inTrain)
train.sample = train.kept8[inTrain,-"SalePrice",with=FALSE]
train.target = train.kept8[inTrain,.(SalePrice=as.numeric(SalePrice))]
test.sample = train.kept8[-inTrain,-"SalePrice",with=FALSE]
test.target = train.kept8[-inTrain,.(SalePrice=as.numeric(SalePrice))]

bootControl <- trainControl(number = 50, verboseIter=TRUE)
tuneGrid = expand.grid(C=c(.25,.5,1),sigma=c(.01,.05,.1,.5))
#bootControl
#tuneGrid
set.seed(6)

print("svmFit = train(x=train.sample,y=train.target$SalePrice,method='svmRadial',tuneLength=5,trControl=bootControl,tuneGrid = tuneGrid, preProcess=c(\"center\",\"scale\"))")
#summary(svmFit)

print("svmFit.prediction=predict(svmFit$finalModel,newdata=test.sample)")

print("rmsle(test.target$SalePrice,svmFit.prediction)")
# sans GardenRatio rmsle=0.144718347370209




