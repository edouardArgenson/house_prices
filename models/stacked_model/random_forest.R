
library('lattice')
library('ggplot2')
library('caret')
library('data.table')
library('Metrics')
library('MASS')
library('e1071')
library('kernlab')
library('ranger')

#print(train)
#print(test)


train = fread('~/kaggle/house_prices/data/train.csv',
              colClasses=c('MiscFeature'='character','PoolQC'='character','Alley'='character'))

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

# MSSubClassCat
train$MSSubClassCat = train[,.(MSSubClassCat=sapply(MSSubClass,toString)),with=TRUE]

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

#which(is.na(train$Exterior2nd))
#table(train$Exterior2nd)
#which(is.na(test$Exterior2nd))

# Separate numerical and categorical features
num_features = names(which(sapply(train,is.numeric)))
cat_features = names(which(sapply(train,is.character)))

num_features
num_features = num_features[-match(c('LotFrontage','OverallQual'),num_features)]
#cat_features

# Select features for training
kept_features = c(num_features,"MSSubClassCat","MSZoning","Neighborhood","HeatingQC","CentralAir","SaleType","SaleCondition",
                 "LotShape","LandContour","BldgType","HouseStyle","Foundation","Exterior1st","Exterior2nd","Functional",
                 "PavedDrive","IsGarage","BsmtCond","BsmtFinType1")
kept_features = kept_features[-1] #Get rid of 'Id'
kept_features = kept_features[-1] #Get rid of 'MSSubClass' which have been replaced by 'MSSubClassCat'
print(kept_features)

# Deal with missing values
LotFrontage_mean = round(mean(train$LotFrontage,na.rm=TRUE))
train[which(is.na(LotFrontage)),'LotFrontage'] <- LotFrontage_mean
train=cbind(train,"IsGarage"=1+numeric(nrow(train)))
train[which(is.na(GarageYrBlt)),'GarageYrBlt'] <- -1
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

# Deal with missing values
test[which(is.na(LotFrontage)),'LotFrontage'] <- LotFrontage_mean
test=cbind(test,"IsGarage"=1+numeric(nrow(test)))
test[which(is.na(GarageYrBlt)),'GarageYrBlt'] <- -1
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

train.kept = train[,kept_features,with=FALSE]
test.kept = test[,kept_features[-which(kept_features=="SalePrice")],with=FALSE]

## For tests on train data
#set.seed(8)
#inTrain = createDataPartition(train.kept$SalePrice,p=.75,list=FALSE)
##print(inTrain)
#train.sample = train.kept[inTrain,-"SalePrice",with=FALSE]
#train.target = train.kept[inTrain,.(SalePrice=as.numeric(SalePrice))]
#test.sample = train.kept[-inTrain,-"SalePrice",with=FALSE]
#test.target = train.kept[-inTrain,.(SalePrice=as.numeric(SalePrice))]

# For predictions for leaderboard
train.sample = train.kept[,-"SalePrice",with=FALSE]
train.target = train.kept[,.(SalePrice=as.numeric(SalePrice))]
#test.sample = scale(test.kept)
test.sample = test.kept

bootControl <- trainControl(number = 20, verboseIter=TRUE)
tuneGrid = expand.grid(mtry=4:25)
#bootControl
#tuneGrid
set.seed(7)

print('rfFit = train(x=train.sample,y=train.target$SalePrice,method=\'ranger\',num.trees=900,min.node.size=4,trControl=bootControl,tuneGrid = tuneGrid, scale=FALSE)')

#0.147776



# get prediction error

#rfFit.prediction = predict(rfFit,newdata=test.sample)
#rmsle(test.target$SalePrice,rfFit.prediction)

# num.trees = 900, mtry = 12, min.node.size = 5 -> rmsle=0.1451406
# num.trees = 900, mtry = 12, min.node.size = 7 -> rmsle=0.1456314

# after full grid search, best min.node.size for 900 trees is min.node.size=4

which(is.na(test.sample$GarageArea))

# leaderboard score 0.15582 =(
