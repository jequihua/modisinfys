import pandas as pd

import numpy as np

from geotiffio import readtif
from geotiffio import createtif
from geotiffio import writetif

import feature_extraction_tools_v2 as fe

# read csv containing all image paths for nadir corrected reflectance produdcts at 1000 m
all_images = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/modis_gpp_1000.csv",\
						 header = 0)