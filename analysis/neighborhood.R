library('gdata')



house_train = read.csv('~/kaggle/house_prices/data/train.csv')

price_per_squareFeet = house_train$SalePrice/house_train$GrLivArea
# hist(price_per_squareFeet,nclass=20)

ngb = table(house_train$Neighborhood)
n_ppsqf = data.frame(neigb=names(ngb),ppsqf=numeric(length(names(ngb))),price=numeric(length(names(ngb))))

compteur = 1
for(i in names(ngb)){
    house_train.temp = subset(house_train,house_train$Neighborhood == i)
    n_ppsqf[compteur,'ppsqf'] = mean(house_train.temp$SalePrice/house_train.temp$GrLivArea)
    n_ppsqf[compteur,'price'] = mean(house_train.temp$SalePrice)
    compteur = compteur + 1
}

print(n_ppsqf)
