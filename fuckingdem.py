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


datasetb,rows,cols,bands = readtif("D:/Julian/64_ie_maps/rasters/covariates/dem1000/2004/dem30_mean1000.tif")
bandb = datasetb.GetRasterBand(1)
bandb = bandb.ReadAsArray(0, 0, cols, rows).astype(np.float64)
bandb = np.ravel(bandb)

nans = np.isnan(bandb)
print(np.sum(nans))

infs = ~np.isfinite(bandb)
print(np.sum(infs))

miss = bandb == -1.70000000e+308
print(np.sum(miss))

infnegs = bandb == "-inf"
print(np.sum(infnegs))

negs = bandb < 0
print(np.sum(negs))

fuckslut = (bandb > -1.70000000e+308) & (bandb < 0)

print(bandb[fuckslut])