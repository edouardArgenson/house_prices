
library('lattice')
library('ggplot2')
library('caret')
library('data.table')

train = fread('~/kaggle/house_prices/data/train.csv',colClasses=c('MiscFeature'='character','PoolQC'='character','Alley'='character'))
test = fread('~/kaggle/house_prices/data/test.csv',colClasses=c('MiscFeature'='character','PoolQC'='character','Alley'='character'))

features = names(train)[-81]
#print(features)

train.sample = train[,features,with=FALSE]
test.sample = test[,features,with=FALSE]

d=train[,.(MSSubClassCat=sapply(MSSubClass,toString)),with=TRUE]
features2=features
features2[2]="MSSubClassCat"
features2=features2[-1]
e = cbind(train.sample,d)
train.sample=e[,features2,with=FALSE]
head(train.sample)

d_test=test[,.(MSSubClassCat=sapply(MSSubClass,toString)),with=TRUE]
e_test = cbind(test.sample,d_test)
test.sample=e_test[,features2,with=FALSE]
head(test.sample)

num_features = names(which(sapply(train.sample,is.numeric)))
cat_features = names(which(sapply(train.sample,is.character)))
#print(num_features)
#print(cat_features)

# Select features for training
kept_features = c(num_features,"Neighborhood","ExterQual","HeatingQC","CentralAir","KitchenQual","SaleType","SaleCondition","IsGarage")
kept_features = kept_features[-7]
print(kept_features)

# Deal with missing values
LotFrontage_mean = round(mean(train$LotFrontage,na.rm=TRUE))
train.sample[which(is.na(LotFrontage)),'LotFrontage'] <- LotFrontage_mean
train.sample=cbind(train.sample,"IsGarage"=1+numeric(nrow(train.sample)))
train.sample[which(is.na(GarageYrBlt)),'GarageYrBlt'] <- 1900
train.sample[which(is.na(GarageQual)),'IsGarage'] <- 0

# Deal with missing values
test.sample[which(is.na(LotFrontage)),'LotFrontage'] <- LotFrontage_mean
test.sample=cbind(test.sample,"IsGarage"=1+numeric(nrow(test.sample)))
test.sample[which(is.na(GarageYrBlt)),'GarageYrBlt'] <- 1900
test.sample[which(is.na(GarageQual)),'IsGarage'] <- 0

train.sample = train.sample[,kept_features,with=FALSE]
test.sample = test.sample[,kept_features,with=FALSE]

#head(train.sample)
#print(names(train.sample))
#print(names(test.sample))
#train.sample[,.(Mean_IsGarage=mean(IsGarage))]
#test.sample[,.(Mean_IsGarage=mean(IsGarage))]

# Prepare target Matrix
train.target=train[,.(SalePrice)]
train.target.toFit = c(sapply(train.target,as.numeric))
head(train.target.toFit)

num_features = c(num_features[-7],"IsGarage")
cat_features=kept_features[35:41]
print(num_features)
print(cat_features)

# Separate numeric and categorical features for conversion
train.sample.num_features = train.sample[,num_features,with=FALSE]
train.sample.cat_features = train.sample[,cat_features,with=FALSE]
test.sample.num_features = test.sample[,num_features,with=FALSE]
test.sample.cat_features = test.sample[,cat_features,with=FALSE]

# Change class of data and merge back numeric and categorical
train.sample.num_features.toFit = train.sample.num_features[,lapply(.SD,as.numeric)]
train.sample.cat_features.toFit = train.sample.cat_features[,lapply(.SD,as.factor)]
train.sample.toFit = cbind(train.sample.num_features.toFit,train.sample.cat_features.toFit)
test.sample.num_features.toPredict = test.sample.num_features[,lapply(.SD,as.numeric)]
test.sample.cat_features.toPredict = test.sample.cat_features[,lapply(.SD,as.factor)]
test.sample.toPredict = cbind(test.sample.num_features.toPredict,test.sample.cat_features.toPredict)

class(train.sample.toFit)
nrow(train.sample.toFit)
ncol(train.sample.toFit)
names(train.sample.toFit)

class(test.sample.toPredict)
nrow(test.sample.toPredict)
ncol(test.sample.toPredict)
names(test.sample.toPredict)

write.table(train.sample.toFit,"~/kaggle/house_prices/models/benchmark_gbm/preprocessed")
write.table(test.sample.toPredict,"~/kaggle/house_prices/models/benchmark_gbm/preprocessed_toPredict")





