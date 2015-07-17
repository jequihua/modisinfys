
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import cross_validation
from sklearn import datasets
from sklearn.metrics import r2_score

from sklearn.externals import joblib

# import training data as data frame
#data = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/training_final_good.csv")
data = pd.read_csv("D:/Julian/64_ie_maps/modelling_20150702/training_tables_finales/final_train_20150716.csv")

# replace -999 flags
data = data.replace("NA",np.nan)


print(type(data))

colnames = data.columns

target_variable=13

print("check target variable")
print(colnames[target_variable])

year_variable=1

data = data.loc\
		[\
		  (data.iloc[:,year_variable] == 0)\
		| (data.iloc[:,year_variable] == 2004)\
		| (data.iloc[:,year_variable] == 2005)\
		| (data.iloc[:,year_variable] == 2006)\
		| (data.iloc[:,year_variable] == 2007)\
		| (data.iloc[:,year_variable] == 2009)\
		| (data.iloc[:,year_variable] == 2010)\
		| (data.iloc[:,year_variable] == 2011)\
		| (data.iloc[:,year_variable] == 2012)\
		| (data.iloc[:,year_variable] == 2013)
		]

#print(np.shape(data))

# data structure
#print(data.head())

# initialize model
rf = RandomForestRegressor(n_estimators=1000,n_jobs=4,max_features=64,min_samples_split=5,oob_score=True)
#rf = ExtraTreesRegressor(n_estimators=1000,n_jobs=4,max_features=30,min_samples_split=5,bootstrap=True,oob_score=True)

# indices of variables of interest (target and covariates)
selection = np.append([target_variable],range(25,216))

# select data of interest
data_selection = data.iloc[:,selection].as_matrix()

# check type of array
#print(np.dtype(data_selection))

# force dtype = float32
data_selection = data_selection.astype(np.float32, copy=False)

# complete cases
goodidx = ~np.isnan(data_selection).any(axis=1)
data_selection = data_selection[goodidx]
#data_selection = data_selection[np.isfinite(data_selection).any(axis=1)]

#np.savetxt("foo.csv",data_selection, delimiter=",")

# target variable / covariates
y = data_selection[:,0]

x = data_selection[:,1:]

# split test-train
#x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.2, random_state=0)

# fit model
pmodel = rf.fit(x,y)
path="D:/Julian/64_ie_maps/modelling_20150702/models/AlturaTotal_media_c1c2_clean/"

joblib.dump(pmodel,path+ 'treeheight20150716.pkl')

# validation measures
oobp = pmodel.oob_prediction_
oobpout = np.zeros((np.shape(data)[0]),dtype=np.float64)
oobpout[oobpout==0]=-1
oobpout[goodidx]=oobp
np.savetxt(path+"0oobprediction20150716.csv", oobpout, delimiter=",")

#oobplog = np.exp(oobp)
corrins = np.corrcoef(oobp,y)
rmse = np.sqrt(np.mean((y - oobp)**2))
mae = np.mean(np.absolute((y - oobp)))
meanofresp = np.mean(y)

# save model

print(corrins)
print(rmse)
print(mae)
print(meanofresp)

statistics = np.array([corrins[0,1],rmse,mae,meanofresp])
np.savetxt(path+"0statistics.csv",statistics, delimiter=",")

# plot

# Print the feature ranking

# column names in searched_data_frame
colnames = data.columns


imp = pmodel.feature_importances_
names = colnames[selection[1:]] 
imp,names = zip(*sorted(zip(imp,names)))
index = range(len(names))
columns = ['variable','importance']
varimpdf = pd.DataFrame(index=index, columns=columns)
varimpdf['variable']=names
varimpdf['importance']=imp
varimpdf = varimpdf.sort(columns="importance",ascending=False)
varimpdf.to_csv(path+"0varimportance20150716.csv", sep=',', encoding='utf-8',index=False)

#print(type(imp))
#imp_selection = imp[0:21]
#names_selection = names[0:21]

#print(type(imp_selection))

#plt.barh(range(21),imp_selection,align="center")
#plt.yticks(range(21),names_selection)

#plt.xlabel("Importance of features")
#plt.ylabel("Features")
#plt.title("Importance of eache feature")
#plt.show()

# print("Feature ranking:")

# for f in range(10):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(10), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(10), indices)
# plt.xlim([-1, 10])
# plt.show()