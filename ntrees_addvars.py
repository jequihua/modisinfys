import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np

from geotiffio import readtif
from geotiffio import createtif
from geotiffio import writetif

import os

from osgeo import gdal

import struct as st

# import training data as data frame
trainff3 = pd.read_csv("D:/Julian/64_ie_maps/cleaning_training/train_ff3_fixed_.csv")
rasterspath = "D:/Julian/64_ie_maps/rasters/climatic/bioclimasneotropicales3/"

# initialize
output = np.zeros((len(trainff3.index),12*3),dtype=np.float64)
layers = []


nfile = 0
for file in os.listdir(rasterspath):
	if file.endswith(".tif"):
		
		layers.append(file)
		print(layers)
		
		data_type = gdal.GDT_Float32
		format ="f"
		
		# read image
		dataset,rows,cols,bands = readtif(rasterspath+file)

		# image metadata
		projection = dataset.GetProjection()
		geotransform = dataset.GetGeoTransform()
		driver = dataset.GetDriver()

		band=dataset.GetRasterBand(1)

 		for i in xrange(len(trainff3.index)):
 			xcoord = trainff3.iloc[i,1]
 			ycoord = trainff3.iloc[i,2]
			xcoord = int((xcoord - geotransform[0]) / geotransform[1])
			ycoord = int((ycoord - geotransform[3]) / geotransform[5])

			# pixel value
			binaryvalue=band.ReadRaster(xcoord,ycoord,1,1,buf_type=data_type)
			value = st.unpack(format, binaryvalue)
			value = value[0]

			output[i,nfile] = value

		nfile=nfile+1

output = pd.DataFrame(output,columns=layers,index=trainff3.index)
output = pd.concat([trainff3,output],axis=1)

# write to disk
output.to_csv("D:/Julian/64_ie_maps/cleaning_training/train_ff3_fixed_angela.csv", sep=',', encoding='utf-8',index=False)
