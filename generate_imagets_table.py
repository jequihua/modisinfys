

import pandas as pd

import numpy as np

from geotiffio import readtif
from geotiffio import createtif
from geotiffio import writetif

import feature_extraction_tools as fe

import gc

# read csv containing all image paths
all_images = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/modis_vegindices_1000.csv", header = 0)

# number of images
nimages = len(all_images.index)

# load filter raster
datasetf,rows,cols,bands = readtif("D:/Julian/64_ie_maps/rasters/filter/bov_cbz_km2.tif")
bandf = datasetf.GetRasterBand(1)
bandf = bandf.ReadAsArray(0, 0, cols, rows).astype(np.int32)
bandf = np.ravel(bandf)

# mexico body mask
gooddatamask = bandf >= 0
ngooddata = np.sum(gooddatamask)
print(ngooddata)

# bad quality values (MODIS QC bits)
badqualityvalues = np.array([-1,2,3,255])

# initialize table testdata
tsdata = np.zeros(((ngooddata),nimages),dtype=np.float32)

for i in xrange(len(all_images.index)):
			
	# read images (variable of interest and associated quality product) 
	dataset,rows,cols,bands = readtif(all_images.iloc[i,5])
					
	# make numpy array and flatten
	band = dataset.GetRasterBand(1)
	band = band.ReadAsArray(0, 0, cols, rows).astype(np.float32)
	band = np.ravel(band)
	
	qdataset,qrows,qcols,qbands = readtif(all_images.iloc[i,7])	
	qband = qdataset.GetRasterBand(1)
	qband = qband.ReadAsArray(0, 0, cols, rows).astype(np.float32)
	qband = np.ravel(qband)

	# check which pixels have bad quality
	qualityaux = np.in1d(qband,badqualityvalues)

	band[qualityaux] = np.nan
	band = band[gooddatamask]
	tsdata[:,i] = band

band = None
qband = None
qdataset = None
dataset = None
bandf = None
datasetf = None

gc.collect()

np.savetxt("ndvi_table.csv",tsdata, delimiter=",")
















# training data csv
#training = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/training_julian_year.csv", header = 0)

# add yeardata
#all_images = fe.generateyeardate(all_images,format="AAAA*MM*DD")
#all_images.to_csv("D:/Julian/64_ie_maps/julian_tables_2/allimages_year.csv", sep=',', encoding='utf-8',index=False)
#training = fe.generateyeardate(training,format="DD*MM*AAAA")
#training.to_csv("D:/Julian/64_ie_maps/julian_tables_2/training_julian_year.csv", sep=',', encoding='utf-8',index=False)

# shape file
#shapefile = "D:/Julian/64_ie_maps/conglomerate/Cong22025_lamb.shp"

# searchdates
#search,lenz = searchdates(training,all_images,positions1=[0,1,2,3,4,5])
# searchdates
#search,lenz = fe.searchdates(training,all_images)

#valuez = extract(1114473.62456,2339981.7047,image="D:/Julian/64_ie_maps/2000_2013_mean_ndvi.tif",data_type=16)

#print(valuez)

#valuez = extract(1114473.62456,2339981.7047,image="D:/Julian/64_ie_maps/2000_2013_mean_ndvi.tif",data_type=32)

#print(valuez)


# query pixels
#tablez = fe.imagestacktable(shapefile=shapefile,data_frame1=training,data_frame2=all_images)

#print(tablez)

#np.savetxt("MIR2.csv", tablez, delimiter=",")




































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

