import pandas as pd

import numpy as np

from geotiffio import readtif
from geotiffio import createtif
from geotiffio import writetif

import feature_extraction_tools_v2 as fe

# read csv containing all image paths for nadir corrected reflectance produdcts at 1000 m
all_images = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/modis_gpp_1000.csv",\
						 header = 0)

# generate date variable
all_images = fe.generatedate(all_images)

variables = np.array([1,3])
fillvalues =np.array([32761,255])

counter = 0
# generate variables
for i in variables:
	print(fillvalues[counter])
	out = fe.sliding_features_months_gpp(all_images,variable=i,quality_variable=None,fillvalue=fillvalues[counter],\
		path="D:/Julian/64_ie_maps/rasters/covariates_month/gpp1000/")
	counter=counter+1


#########################################################################################################################################
