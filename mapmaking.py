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

from geotiffio import readtif
from geotiffio import createtif
from geotiffio import writetif

import feature_extraction_tools as fe

# load model
path = "D:/Julian/64_ie_maps/models/average_tree_height/1000m/random_forest/ff3_rf_1000_85_5/"
pmodel = joblib.load(path+'ff3_rf_1000_85_5.pkl')
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

	fe.save_file(dataset,predictionout, rows, cols, path="D:/Julian/64_ie_maps/rasters/products/aheight_ff3_rf_1000_85_5/", base_date=year, varname="treeheight", sufix="mean")