import pandas as pd

import numpy as np

from geotiffio import readtif
from geotiffio import createtif
from geotiffio import writetif

import feature_extraction_tools_v2 as fe

# # read csv containing all image paths for nadir corrected reflectance produdcts at 1000 m
# all_images = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/modis_nadirreflectance_1000.csv",\
# 						 header = 0)

# # generate date variable
# all_images = fe.generatedate(all_images)

# variables = np.array([1,2,3,4,5,6,7])

# # generate variables
# for i in variables:
# 	out = fe.sliding_features_months(all_images,variable=i,quality_variable=None,fillvalue=32767,path="D:/Julian/64_ie_maps/rasters/covariates_month/nreflectance1000/")


#########################################################################################################################################


# read csv containing all image paths for vegetation indices at 1000 m
all_images = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/modis_lai_1000.csv",\
						 header = 0)

# # generate date variable
all_images = fe.generatedate(all_images)

variables = np.array([1,5])

# generate variables
counter=0
for i in variables:
	
	out = fe.sliding_features(all_images,variable=i,fillvalue=249,path="D:/Julian/capas_patybalvanera/LAI_1km/")
	counter = counter+1
