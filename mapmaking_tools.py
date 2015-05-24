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

# # base raster (DEM raster)
# datasetb,rows,cols,bands = readtif("D:/Julian/64_ie_maps/rasters/covariates/dem1000/2004/dem30_mean1000.tif")
# bandb = datasetb.GetRasterBand(1)
# bandb = bandb.ReadAsArray(0, 0, cols, rows).astype(np.float64)
# bandb = np.ravel(bandb)
# #bandb[bandb==-1.70000000e+308] = np.nan

# # filter raster
# datasetf,rows,cols,bands = readtif("D:/Julian/64_ie_maps/rasters/filter/bov_cbz_km2.tif")
# bandf = datasetf.GetRasterBand(1)
# bandf = bandf.ReadAsArray(0, 0, cols, rows).astype(np.float64)
# bandf = np.ravel(bandf)
# print(bandf)

# # remove missings and infs etc
# maskmissings = bandb == -1.70000000e+308
# goodbandbmean= np.mean(bandb[~maskmissings])
# maskfalsemissings = bandf >= 0
# mask = maskmissings & maskfalsemissings

# # fill in gaps 
# print(goodbandbmean)
# bandb[mask] = goodbandbmean
# fe.save_file(datasetb,bandb, rows, cols, path="D:/Julian/64_ie_maps/rasters/", base_date=year, varname="dem", sufix="clean")