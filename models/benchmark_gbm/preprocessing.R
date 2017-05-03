library('caret')
library('data.table')

train = fread('~/kaggle/house_prices/data/train.csv',colClasses=c('MiscFeature'='character','PoolQC'='character'))

features = names(train)[-81]
#print(features)

train.sample = train[,features,with=FALSE]

d=train[,.(MSSubClassCat=sapply(MSSubClass,toString)),with=TRUE]
features2=features
features2[2]="MSSubClassCat"
features2=features2[-1]
e = cbind(train.sample,d)
train.sample=e[,features2,with=FALSE]
#head(train.sample)

num_features = names(which(sapply(train.sample,is.numeric)))
cat_features = names(which(sapply(train.sample,is.character)))
#print(num_features)
#print(cat_features)

kept_features = c(num_features,"Neighborhood","ExterQual","HeatingQC","CentralAir","KitchenQual","SaleType","SaleCondition","IsGarage")
kept_features = kept_features[-7]
#print(kept_features)

LotFrontage_mean = round(mean(train$LotFrontage,na.rm=TRUE))
train.sample[which(is.na(LotFrontage)),'LotFrontage'] <- LotFrontage_mean
train.sample=cbind(train.sample,"IsGarage"=1+numeric(nrow(train.sample)))
train.sample[which(is.na(GarageYrBlt)),'GarageYrBlt'] <- 1900
train.sample[which(is.na(GarageQual)),'IsGarage'] <- 0

train.sample = train.sample[,kept_features,with=FALSE]

#head(train.sample)
#print(names(train.sample))
#train.sample[,.(Mean_IsGarage=mean(IsGarage))]

train.target=train[,.(SalePrice)]
train.target.toFit = c(sapply(train.target,as.numeric))
#head(train.target.toFit)

num_features = c(num_features[-7],"IsGarage")
cat_features=kept_features[35:41]
#print(num_features)
#print(cat_features)

train.sample.num_features = train.sample[,num_features,with=FALSE]
train.sample.cat_features = train.sample[,cat_features,with=FALSE]

train.sample.num_features.toFit = train.sample.num_features[,lapply(.SD,as.numeric)]
train.sample.cat_features.toFit = train.sample.cat_features[,lapply(.SD,as.factor)]
train.sample.toFit = cbind(train.sample.num_features.toFit,train.sample.cat_features.toFit)

#class(train.sample.toFit)
#nrow(train.sample.toFit)
#ncol(train.sample.toFit)
#names(train.sample.toFit)

#write.table(train.sample.toFit,"~/kaggle/house_prices/models/benchmark_gbm/preprocessed")






