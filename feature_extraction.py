import pandas as pd

import numpy as np

from geotiffio import readtif
from geotiffio import createtif
from geotiffio import writetif

import feature_extraction_tools as fe

# read csv containing all image paths
all_images = pd.read_csv("/home/jequihua/Documents/analisis_robin/inputs/modis_temperature_1000.csv", header = 0)
all_images = fe.generatedate(all_images)
all_images.to_csv("/home/jequihua/Documents/analisis_robin/inputs/modis_temperature_1000_year.csv", sep=',', encoding='utf-8',index=False)

# subsetz

variables = [7,8]
qvariables = [11,12]
#fillvalues = [-1000,-3000,-1000,-3000,-1000,-1000]
counter = 0

for i in variables:
	subset = fe.sliding_features(all_images,path="/home/jequihua/Documents/analisis_robin/outputs/rasters/",variable=i,quality_variable=qvariables[counter])
	counter = counter+1

