# Pre-traitement pour supprimer/remplacer les valeurs NA dans les features de type 'numeric'
# Convertit egalement la feature 'MSSubClassCat' en variable categorique


library('caret')
library('data.table')

# Charger les donnes
train = fread('~/kaggle/house_prices/data/train.csv',colClasses=c('MiscFeature'='character','PoolQC'='character'))

features = names(train)[-81]
train.sample = train[,features,with=FALSE]

# 1. Convertir 'MSSubClassCat' en variable categorique
d=train[,.(MSSubClassCat=sapply(MSSubClass,toString)),with=TRUE]
features2=features
features2[2]="MSSubClassCat"
e = cbind(train.sample,d)
train.sample=e[,features2,with=FALSE]

num_features = which(sapply(train.sample,is.numeric))
cat_features = which(sapply(train.sample,is.character))

# Enlever la colonne 'Id'
num_features=num_features[-1]

train.sample.cat_features = train.sample[,cat_features,with=FALSE]
train.sample.num_features = train.sample[,num_features,with=FALSE]

# 2. Remplacer les valeurs manquantes dans 'LotFrontage' par la moyenne
LotFrontage_mean = round(mean(train$LotFrontage,na.rm=TRUE))
train.sample[which(is.na(LotFrontage)),'LotFrontage'] <- LotFrontage_mean

# 3. Creer une colonne IsGarage qui vaut 1 si la maison comporte un garage, 0 sinon. Dans 'GarageYrBlt' remplacer 'NA' par 1900 = min(GarageYrBlt) si pas de garage.
train.sample=cbind(train.sample,"IsGarage"=1+numeric(nrow(train.sample)))
train.sample[which(is.na(GarageYrBlt)),'GarageYrBlt'] <- 1900
train.sample[which(is.na(GarageQual)),'IsGarage'] <- 0

## 4. Enlever les 8 lignes pour lesquelles 'MasVnrArea'=='NA'
## Finalement on les garde
## Pour l instant on choisit de ne pas utiliser cet attribut
#train.sample = train.sample[-which(is.na(MasVnrArea)),]
