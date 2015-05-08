import numpy as np

import pandas as pd

from geotiffio import readtif
from geotiffio import createtif
from geotiffio import writetif

import jdcal as jd

from osgeo import gdal
from osgeo import ogr

import os

import struct as st

# v = np.array([[1,2.45,3,4,5,6,7,8,9,10,100,100,100,99.1,99.2,99.2],\
# 			 [12,2.454,34,45,54,63,72,81,91,1,100,100,100,99.1,99.2,99.2],\
# 			 [1123,23.4215,3,4,5,6,7,8,92,10,100,100,100,9.1,9.2,9.2]],dtype=np.float64)

# print(np.mean(v,axis=0))

# mask = v == 100

# #print(mask)

# vm = np.ma.array(v,mask=mask)

# print(np.mean(vm,axis=0))
# print(np.ma.mean(vm,axis=0))

v= np.array([1,2,3,4,5,6,7,8,9])

#what = np.multiply(v,v)

aux = np.in1d(v, [1,3,9])

print(aux)

print(np.sum(aux))

