# Premier modele pour faire un benchmark.
# Regression lineaire sur 5 features:
# "OverallQual"  "GrLivArea"  "FullBath"  "YearBuilt"  "BedroomAbvGr"

# Resultats: RMSLE=0.23014 



library('caret')
library('data.table')
library('Metrics')

train = fread('~/kaggle/house_prices/data/train.csv',colClasses=c('MiscFeature'='character','PoolQC'='character'))


kept_features = c("OverallQual","GrLivArea","FullBath","YearBuilt","BedroomAbvGr")

train.sample = train[,c(kept_features,"SalePrice"),with=FALSE]


# Pas utilise par erreur, a utiliser
bootControl = trainControl(number=25)

# On fit le modele
lmFit = train(SalePrice ~ 1+OverallQual+GrLivArea+FullBath+YearBuilt+BedroomAbvGr,method='lm',data=train.sample,metric="RMSE")

# chargement des donnees a predire
test.sample_submission = fread('~/kaggle/house_prices/data/sample_submission.csv')
test.sample_submission = test.sample_submission[,.(Id)]
test.sample = fread('~/kaggle/house_prices/data/test.csv',colClasses=c('MiscFeature'='character','PoolQC'='character','Alley'='character'))
test.sample = test.sample[,kept_features,with=FALSE]

# prediction
lmFit.prediction=predict(lmFit$finalModel,newdata=test.sample)

# ecriture des resultats
test.sample_submission.new = cbind(test.sample_submission,lmFit.prediction)
test.sample_submission = test.sample_submission.new[,.(Id,SalePrice=lmFit.prediction)]
write.csv(test.sample_submission,'~/kaggle/house_prices/data/my_submission.csv',row.names=FALSE)
