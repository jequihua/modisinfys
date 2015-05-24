
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

import os

import gc

# import training data as data frame
#data = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/training_final_good.csv")
data = pd.read_csv("D:/Julian/64_ie_maps/cleaning_training/train_ff3.csv")


# replace -999 flags
#data = data.replace("NA",np.nan)

# 16 models must be made
variables = [260,261,262,263,265,266,267,268,270,271,272,273,275,276,277,278]

colnames = data.columns
year_variable=284
print("check year variable")
print(colnames[year_variable])

for i in range(len(variables)):

	target_variable=variables[i]
	print("check target variable")
	varname =colnames[target_variable]
	print(varname)

	# initialize model
	rf = RandomForestRegressor(n_estimators=1000,n_jobs=4,max_features=85,min_samples_split=5,oob_score=True)
	#rf = ExtraTreesRegressor(n_estimators=1000,n_jobs=4,max_features=30,min_samples_split=5,bootstrap=True,oob_score=True)

	# indices of variables of interest (target and covariates)
	selection = np.append([target_variable],range(4,258))
	selection = np.append(selection,[year_variable])

	# select data of interest
	data_selection = data.iloc[:,selection].as_matrix()

	# check type of array
	#print(np.dtype(data_selection))

	# force dtype = float32
	#data_selection = data_selection.astype(np.float32, copy=False)

	# complete cases
	data_selection = data_selection[~np.isnan(data_selection).any(axis=1)]
	data_selection = data_selection[np.isfinite(data_selection).any(axis=1)]

	#np.savetxt("foo.csv",data_selection, delimiter=",")

	# target variable / covariates
	y = data_selection[:,0].astype(np.float64)
	x = data_selection[:,1:]

	# split test-train
	#x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.2, random_state=0)

	# fit model
	pmodel = rf.fit(x,y)
	path = "D:/Julian/64_ie_maps/models/"+varname+"/"
	if not os.path.exists(path):
		os.makedirs(path)
	#path="D:/Julian/64_ie_maps/models/average_tree_height/1000m/random_forest/ff3_rf_1000_85_5/"

	joblib.dump(pmodel,path+varname+'_ff3_rf_1000_85_5.pkl')

	# validation measures
	oobp = pmodel.oob_prediction_
	oobplog = np.exp(oobp)
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
	np.savetxt(path+"0"+varname+"statistics.csv",statistics, delimiter=",")

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
	varimpdf.to_csv(path+"0varimp_ff3_rf_1000_85_5.csv", sep=',', encoding='utf-8',index=False)

gc.collect()

# import training data as data frame
#data = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/training_final_good.csv")
data = pd.read_csv("D:/Julian/64_ie_maps/cleaning_training/train_ff3.csv")

# 16 maps must be made
variables = [260,261,262,263,265,266,267,268,270,271,272,273,275,276,277,278]

for i in range(len(variables)):

	target_variable=variables[i]
	print("check target variable")
	varname =colnames[target_variable]
	print(varname)

	# load model
	path = "D:/Julian/64_ie_maps/models/"+varname+"/"
	pmodel = joblib.load(path+varname+'_ff3_rf_1000_85_5.pkl')
	print(type(pmodel))

	# load variable selection list:
	imagesdf = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/covariatesmodels_cropped_85.csv", header = 0) 


	# # filter raster
	datasetf,rows,cols,bands = readtif("D:/Julian/64_ie_maps/rasters/filter/bov_cbz_km2.tif")
	bandf = datasetf.GetRasterBand(1)
	bandf = bandf.ReadAsArray(0, 0, cols, rows).astype(np.float64)
	bandf = np.ravel(bandf)

	# mexico body mask
	baddatamask = bandf < 0

	# testdata
	nvar = int(len(imagesdf.index))+1
	testdata = np.zeros(((cols*rows),nvar),dtype=np.float64)

	for y in xrange(10):
		year=2004+y
		print(year)
		for i in xrange(len(imagesdf.index)):
			
			# read images (variable of interest and associated quality product) 
			imagesdf.columns[2+y]
			dataset,rows,cols,bands = readtif(imagesdf.iloc[i,2+y])
					
			# make numpy array and flatten
			band = dataset.GetRasterBand(1)
			band = band.ReadAsArray(0, 0, cols, rows).astype(np.float64)
			band = np.ravel(band)
			if (imagesdf.iloc[i,1]=="demmean") | (imagesdf.iloc[i,1]=="demsd"):
				maskmissings = (band == -1.70000000e+308)
				goodbandmean= np.mean(band[~maskmissings])
				mask = maskmissings & (~baddatamask)
				band[mask] = goodbandmean
			band[baddatamask] = np.nan
			testdata[:,i] = band

		testdata[:,nvar-1]=year

		# remove empty cells
		goodidx = ~np.isnan(testdata[:,0])
		data = testdata[goodidx,:]

		# fill in 0's
		#for j in xrange(np.shape(data)[1]):
		#	mzero = data[:,j]==0
		#	data[mzero,j]=np.mean(data[mzero,j])

		# prediction
		print("predicting at last")
		prediction = pmodel.predict(data)

		predictionout = np.zeros((cols * rows),dtype=np.float64)
		predictionout[predictionout==0]=-999

		predictionout[goodidx]=prediction
		outpath = "D:/Julian/64_ie_maps/rasters/products/"+varname+"/"
		if not os.path.exists(outpath):
			os.makedirs(outpath)
		fe.save_file(dataset,predictionout, rows, cols, path=outpath, base_date=year, varname=varname, sufix="_")