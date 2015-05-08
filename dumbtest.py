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

v = np.array([[1,2.45,3,4,5,6,7,8,9,10,100,100,100,99.1,99.2,99.2],\
 			 [12,2.454,34,45,54,63,72,81,91,1,100,100,100,99.1,99.2,99.2],\
			 [1123,23.4215,3,4,5,6,7,8,92,10,100,100,100,9.1,9.2,9.2]],dtype=np.float64)


v[v == 100]=np.nan

p = np.nanpercentile(v,50,axis=1)

print(p)

print(np.median(v,axis=1))

