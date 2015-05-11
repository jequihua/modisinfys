import pandas as pd

import numpy as np

from geotiffio import readtif
from geotiffio import createtif
from geotiffio import writetif

import feature_extraction_tools as fe

# read csv containing all image paths
all_images = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/allimages_year2.csv", header = 0)
#all_images = fe.generatedate(all_images)
#all_images.to_csv("D:/Julian/64_ie_maps/julian_tables_2/allimages_year2.csv", sep=',', encoding='utf-8',index=False)

# subsetz

variables = [1,3,4,5,6,8]
fillvalues = [-1000,-3000,-1000,-3000,-1000,-1000]
counter = 0

for i in variables:
	subset = fe.sliding_features(all_images,path="D:/Julian/64_ie_maps/rasters/covariates/",variable=i,fillvalue=fillvalues[counter])
	
