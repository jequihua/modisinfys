

import pandas as pd

import numpy as np

from geotiffio import readtif
from geotiffio import createtif
from geotiffio import writetif

import feature_extraction_tools as fe

# read csv containing all image paths
all_images = pd.read_csv("/home/jequihua/Documents/analisis_robin/scripts/allimages_julian.csv", header = 0)
all_images = fe.generateyeardate(all_images)

# training data csv
training = pd.read_csv("/home/jequihua/Documents/analisis_robin/scripts/newest/training_julian_year.csv", header = 0)

# shape file
shapefile = "/home/jequihua/Documents/analisis_robin/scripts/shape/Cong22025_lamb.shp"

# variable lists
variables = [1,3,4,5,6,8]
variable_names = ["blue_2year.csv","EVI_2year.csv","MIR_2year.csv","NDVI_2year.csv","NIR_2year.csv","red_2year.csv"]
fillvalues =[-1000,-3000,-1000,-3000,-1000,-1000]

counter = 0
for i in variables:
	tablez = fe.imagestacktable(shapefile=shapefile,data_frame1=training,data_frame2=all_images,\
						variable=i,quality_variable=7,fillvalue=fillvalues[counter])
	np.savetxt(variable_names[counter], tablez, delimiter=",")
	counter = counter+1



































# shapefile
#shp = readshape("D:/Julian/64_ie_maps/conglomerate/Cong22025_lamb.shp")
#shp = shapecoordinates("Cong22025_lamb.shp")

#merged = mergedataframes(shp,training)

#merged.to_csv("merge_test.csv", sep=',', encoding='utf-8',index=False)

# jd.gcal2jd
#print(jd.gcal2jd(2000, 1, 1))
#print(jd.gcal2jd(2000, 1, 2))

#print(search.at[0,'NDVI'])

# read first image to get metadata
#dataset,rows,cols,bands = readtif(search.at[0,'NDVI'])

# save characteristics of the images to be assiged to the output image
#projection = dataset.GetProjection()
#transform = dataset.GetGeoTransform()
#driver = dataset.GetDriver()





# # initialize huge vector
# image_time_series = np.zeros((len(all_images.index), cols * rows))

# for i in xrange(len(all_images.index)):
# #for i in xrange(10):
	
# 	# read envi image i
# 	dataset,rows,cols,bands = readtif(all_images.at[i,'NDVI'])
	
# 	# make numpy array and flatten
# 	band = dataset.GetRasterBand(1)
# 	band = band.ReadAsArray(0, 0, cols, rows).astype(float)
# 	image_time_series[i, :] = np.ma.array(np.ravel(band),mask=-3000)
# 	#image_time_series[i, :] = np.ma.MaskedArray(np.ravel(band),mask=(-3000))
# 	#print(image_time_series[i, :])
# 	#image_time_series[i, :] = np.ravel(band)

# column_means = np.ma.mean(image_time_series,axis=0)
# #column_means = np.mean(image_time_series,axis=0)

# outData = createtif(driver,rows,cols,1,"2000_2013_mean_ndvi_MaskedArray_2.tif")
# writetif(outData,column_means,projection,transform,order='r')
