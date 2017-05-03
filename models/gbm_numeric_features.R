library('caret')
library('data.table')
train = fread('~/kaggle/house_prices/data/train.csv',colClasses=c('MiscFeature'='character','PoolQC'='character'))
features = names(train)[-81]
train.sample = train[,features,with=FALSE]
d=train[,.(MSSubClassCat=sapply(MSSubClass,toString)),with=TRUE]
features2=features
features2[2]="MSSubClassCat"
e = cbind(train.sample,d)
train.sample=e[,features2,with=FALSE]
num_features = which(sapply(train.sample,is.numeric))
cat_features = which(sapply(train.sample,is.character))
num_features=num_features[-1]
train.sample.cat_features = train.sample[,cat_features,with=FALSE]
train.sample.num_features = train.sample[,num_features,with=FALSE]
LotFrontage_mean = round(mean(train$LotFrontage,na.rm=TRUE))
train.sample[which(is.na(LotFrontage)),'LotFrontage'] <- LotFrontage_mean
train.sample=cbind(train.sample,"IsGarage"=1+numeric(nrow(train.sample)))
bootControl = trainControl(number=25)
train.target=train[,.(SalePrice)]
bootControl = trainControl(number=25,verboseIter=TRUE)
train.sample[which(is.na(GarageYrBlt)),'GarageYrBlt'] <- 1900
train.sample[which(is.na(GarageQual)),'IsGarage'] <- 0
num_features=num_features[-7]
train.sample.num_features = train.sample[,num_features,with=FALSE]
gbmGrid = expand.grid(interaction.depth = 2*(2:5),n.trees = (4:10)*25,shrinkage=.1,n.minobsinnode=10)

train.target.toFit = c(sapply(train.target,as.numeric))

inTrain = createDataPartition(train.target.toFit,p=3/4,list=FALSE)
num_features = which(sapply(train.sample,is.numeric))
num_features=num_features[-1]
train.sample.num_features = train.sample[,num_features,with=FALSE]

train.sample.toFit = train.sample.num_features[,lapply(.SD,as.numeric)]
train.target.toFit = train.target[,lapply(.SD,as.numeric)]

train.sample.toFit.train = train.sample.toFit[inTrain,]
train.sample.toFit.valid = train.sample.toFit[-inTrain,]

train.target.toFit = c(sapply(train.target,as.numeric))
train.target.toFit.train = train.target.toFit[inTrain]
train.target.toFit.valid = train.target.toFit[-inTrain]

gbmFit = train(train.sample.toFit.train,train.target.toFit.train,method='gbm',trControl=bootControl,verbose=TRUE,bag.fraction=.5,tuneGrid=gbmGrid,metric='RMSE')

train.target.predicted = predict(gbmFit$finalModel,newdata=train.sample.toFit.valid,n.trees=250)
rmsle(train.target.toFit.valid,train.target.predicted)

gbmGrid2 = expand.grid(interaction.depth = 2*(1:5),n.trees = (4:20)*25,shrinkage=.1,n.minobsinnode=10)

gbmFit2 = train(train.sample.toFit.train,train.target.toFit.train,method='gbm',trControl=bootControl,verbose=TRUE,bag.fraction=.5,tuneGrid=gbmGrid2,metric='RMSE')
train.target.predicted2 = predict(gbmFit2$finalModel,newdata=train.sample.toFit.valid,n.trees=475)
rmsle(train.target.toFit.valid,train.target.predicted2)



