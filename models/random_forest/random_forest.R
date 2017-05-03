
library('lattice')
library('ggplot2')
library('caret')
library('data.table')
library('Metrics')
library('MASS')
library('e1071')
library('kernlab')
library('ranger')

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

train$MSSubClassCat = train[,.(MSSubClassCat=sapply(MSSubClass,toString)),with=TRUE]

# Separate numerical and categorical features
num_features = names(which(sapply(train,is.numeric)))
cat_features = names(which(sapply(train,is.character)))

#num_features
#cat_features

# Select features for training
kept_features = c(num_features,"MSSubClassCat","MSZoning","Neighborhood","HeatingQC","CentralAir","SaleCondition",
                  "IsGarage","LotShape","LandContour","BldgType","HouseStyle","Foundation","Exterior1st","SaleType",
                  "Exterior2nd","Functional","PavedDrive","BsmtExposure","FireplaceQu","GarageType","GarageQual",
                 "GarageCond","BsmtCond",'BsmtFinType1',"Fence","MasVnrType","Alley","LandSlope","Condition1",
                 "RoofStyle","Electrical")

# Get rid of a few features
rejected=c("LotFrontage","BsmtFinSF2","LowQualFinSF","BsmtHalfBath",
                   "ThreeSsnPorch","MiscVal",
                   "MoSold","YrSold","Id","MSSubClass")
kept_features = kept_features[-match(rejected,num_features)]

print(kept_features)

# Deal with missing values
LotFrontage_mean = round(mean(train$LotFrontage,na.rm=TRUE))
train[which(is.na(LotFrontage)),'LotFrontage'] <- LotFrontage_mean
train=cbind(train,"IsGarage"=1+numeric(nrow(train)))
train[which(is.na(GarageYrBlt)),'GarageYrBlt'] <- -1
train[which(is.na(GarageQual)),'IsGarage'] <- 0
train[which(is.na(MasVnrArea)),'MasVnrArea'] <- 0
train[which(is.na(BsmtCond)),'BsmtCond'] <- 'MISSING'
train[which(is.na(BsmtFinType1)),'BsmtFinType1'] <- 'MISSING'
train[which(is.na(BsmtExposure)),'BsmtExposure'] <- 'MISSING'
train[which(is.na(FireplaceQu)),'FireplaceQu'] <- 'MISSING'
train[which(is.na(GarageType)),'GarageType'] <- 'MISSING'
train[which(is.na(GarageQual)),'GarageQual'] <- 'MISSING'
train[which(is.na(GarageCond)),'GarageCond'] <- 'MISSING'
train[which(is.na(Fence)),'Fence'] <- 'MISSING'
train[which(is.na(MasVnrType)),'MasVnrType'] <- 'MISSING'
train[which(is.na(Alley)),'Alley'] <- 'MISSING'
train[which(is.na(Condition1)),'Condition1'] <- 'MISSING'
train[which(is.na(Electrical)),'Electrical'] <- 'MISSING'

## Deal with missing values
#test.sample[which(is.na(LotFrontage)),'LotFrontage'] <- LotFrontage_mean
#test.sample=cbind(test.sample,"IsGarage"=1+numeric(nrow(test.sample)))
#test.sample[which(is.na(GarageYrBlt)),'GarageYrBlt'] <- 1900
#test.sample[which(is.na(GarageQual)),'IsGarage'] <- 0
#test[which.is.na(MasVnrArea)),'MasVnrArea'] <- 0

train.kept = train[,kept_features,with=FALSE]

# For tests on train data
set.seed(8)
inTrain = createDataPartition(train.kept$SalePrice,p=.75,list=FALSE)
#print(inTrain)
train.sample = train.kept[inTrain,-"SalePrice",with=FALSE]
train.target = train.kept[inTrain,.(SalePrice=as.numeric(SalePrice))]
test.sample = train.kept[-inTrain,-"SalePrice",with=FALSE]
test.target = train.kept[-inTrain,.(SalePrice=as.numeric(SalePrice))]

## For predictions for leaderboard
#train.sample = train.kept[,-"SalePrice",with=FALSE]
#train.target = train.kept[,.(SalePrice=as.numeric(SalePrice))]
#test.sample = scale(test.kept)
##test.target = train.kept8[-inTrain,.(SalePrice=as.numeric(SalePrice))]

bootControl <- trainControl(number = 30, verboseIter=TRUE)
tuneGrid = expand.grid(mtry=8:26)
#bootControl
#tuneGrid
set.seed(7)

print('rfFit = train(x=train.sample,y=train.target$SalePrice,method=\'ranger\',num.trees=900,trControl=bootControl,tuneGrid = tuneGrid, scale=FALSE)')

# 500 -> 0.1458119
# 700 -> 0.1447977
# 900 -> 0.1446978
# 900 without few feats -> 0.1442677
# 1200 wff -> 0.1450779

