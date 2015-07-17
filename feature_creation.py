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


# # read csv containing all image paths for vegetation indices at 1000 m
# all_images = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/modis_vegindices_1000_year.csv",\
# 						 header = 0)


# variables = np.array([6])
# fillvalues = [-1000]

# # generate variables
# counter=0
# for i in variables:
	
# 	out = fe.sliding_features_months(all_images,variable=i,fillvalue=fillvalues[counter],path="D:/Julian/64_ie_maps/rasters/covariates_month/vegindices1000/")
# 	counter = counter+1

############################################################################################

# read csv containing all image paths
training = pd.read_csv("D:/Julian/64_ie_maps/modelling_20150702/training_tables_finales/vcf10_train600_20150708.csv",\
						 header = 0)

# all_images csv
all_images = pd.read_csv("D:/Julian/64_ie_maps/modelling_20150702/covariate_tables/covariates20150702.csv",\
						 header = 0)

# associate pixel-values <-> in situ measurements
date,table = fe.noshpimagestacktable(data_frame1=training,data_frame2=all_images)

print(date)

# write to disk
table.to_csv("D:/Julian/64_ie_maps/modelling_20150702/training_tables_finales/vcf10_train600_20150708_cov.csv", sep=',', encoding='utf-8',index=False)
