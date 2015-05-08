
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


def tojulian(date_string,format):

	'''
	converts a string into a julian date number 
	
	'''

	if format == "AAAA*MM*DD":
		year = date_string[0:4]
		month = date_string[5:7]
		day = date_string[8:10]
		juliandate = jd.gcal2jd(year,month,day)[1]
	
	elif format == "DD*MM*AAAA":
		year = date_string[6:10]
		month = date_string[3:5]
		day = date_string[0:2]
		juliandate = jd.gcal2jd(year,month,day)[1]

	return juliandate

def generatejuliandate(data_frame,positions=None,date_variable="date",\
	format="AAAA*MM*DD"):
	
	'''
	add a julian date column to a dataframe

	dataframe must have an existing date column

	the format of the original date column must be AAAA*MM*DD or DD*MM*AAAA

	output can be a (row-wise) subset of the original
	
	'''

	if positions is None:
		positions = range(len(data_frame.index))

	n = len(positions)
	# initialize
	juliandates = np.ones((n))

	for i in positions:
		juliandates[i]=float(tojulian(data_frame.loc[i,str(date_variable)],\
			format=format))

def generateyeardate(data_frame,positions=None,date_variable="date",\
	format="AAAA*MM*DD"):
	
	'''
	add a year date column to a dataframe

	dataframe must have an existing date column

	the format of the original date column must be AAAA*MM*DD or DD*MM*AAAA

	output can be a (row-wise) subset of the original
	
	'''

	if positions is None:
		positions = range(len(data_frame.index))

	n = len(positions)
	# initialize
	yeardates = np.ones((n))

	for i in positions:
			if format == "AAAA*MM*DD":
				year = data_frame.loc[i,str(date_variable)]
				year = year[0:4]
	
			elif format == "DD*MM*AAAA":
				year = data_frame.loc[i,str(date_variable)]
				year = year[6:10]
			yeardates[i]=float(year)

	data_frame = data_frame.iloc[positions,:]
	data_frame['yeardate'] = pd.Series(yeardates, index=data_frame.index)
	return data_frame

def mergedataframes(dataframe1,dataframe2,on="IdConglome",howtomerge="outer",\
	notnull=True,howtodrop="any"):

	merged = pd.merge(dataframe1, dataframe2, on=on, how=howtomerge)

	if notnull:
		merged = merged.dropna(how=howtodrop)
	return merged 

def shapecoordinates(shape,output="dataframe",field="IdConglome"):
	'''
	Extract coordinates of each object of a shapefile
	and return as pandas dataframe or numpy array

	'''
	driver = ogr.GetDriverByName('ESRI Shapefile')
	ds=driver.Open(shape,0)
	layer = ds.GetLayer()

	# number of objects in shapefile
	count = layer.GetFeatureCount()

	# initialize numpy array to fill with objectid, xcoord and ycoord
	coordinates = np.zeros((3,count))

	for i in xrange(count):

		feature = layer.GetFeature(i)
		geom = feature.GetGeometryRef()
		mx,my = geom.GetX(), geom.GetY()
		
		coordinates[0,i] = feature.GetField(field)
		coordinates[1,i] = mx
		coordinates[2,i] = my


	if output=="dataframe":
		index = range(count)
		columns = ["IdConglome","x","y"]
		coordinates=pd.DataFrame.from_records(data=coordinates[[0,1,2],:].T,index=index,columns=columns)
	
	return coordinates

	# flush memory
	feature.Destroy()
	ds.Destroy()


def searchdates(data_frame1,data_frame2,date_variable1="yeardate",date_variable2="yeardate",days=None, \
	years=1,positions1=None,positions2=None):

	'''
	subset data frame with another data frame based on a 
	# of days neighborhood: [base_date - days, base_date + days]	
	or a certain year + years

	'''
	if positions1 is not None:
		data_frame1 = data_frame1.loc[positions1,:]

	if positions2 is not None:
		data_frame2 = data_frame2.loc[positions2,:]

	# column names in search_data_frame
	names1 = list(data_frame1.columns.values)
	
	# finde index of date variable
	idx_names1 = names1.index(date_variable1)

	# column names in searched_data_frame
	names2 = list(data_frame2.columns.values)
	
	# finde index of date variable
	idx_names2 = names2.index(date_variable2)

	# number of rows in search_data_frame
	nrows1 = len(data_frame1.index)

	data_frames = []
	data_length = []
	
	if days is not None:
		for i in xrange(nrows1):

			base_date = float(data_frame1.iloc[[i],idx_names1])

			searched_data_frame = data_frame2.loc[(data_frame2.iloc[:,idx_names2] <= (base_date + days)) \
			& (data_frame2.iloc[:,idx_names2] >= (base_date - days)) ]

			data_frames.append(searched_data_frame)
			data_length.append(len(searched_data_frame.index))
		
		return data_frames, data_length
	else:
		for i in xrange(nrows1):

			base_date = float(data_frame1.iloc[[i],idx_names1])

			searched_data_frame = data_frame2.loc[(data_frame2.iloc[:,idx_names2] == (base_date - years))\
			| (data_frame2.iloc[:,idx_names2] == (base_date))\
			| (data_frame2.iloc[:,idx_names2] == (base_date + years))]

			data_frames.append(searched_data_frame)
			data_length.append(len(searched_data_frame.index))
		
		return data_frames, data_length


def extract(xcoord,ycoord,image,qualityimage=None,fillvalue=-3000,\
					badqualityvalues=np.array([-1,2,3]),data_type=32):

	'''
	extract raster-value at a given location	

	'''
	if data_type == 32:
		data_type = gdal.GDT_Float32
		format ="f"
	else:
		data_type = gdal.GDT_Int16
		format ="h"


	# read image
	dataset,rows,cols,bands = readtif(image)

	# image metadata
	projection = dataset.GetProjection()
	geotransform = dataset.GetGeoTransform()
	driver = dataset.GetDriver()

	xcoord = int((xcoord - geotransform[0]) / geotransform[1])
	ycoord = int((ycoord - geotransform[3]) / geotransform[5])

	# pixel value
	band=dataset.GetRasterBand(1)
	binaryvalue=band.ReadRaster(xcoord,ycoord,1,1,buf_type=data_type)
	value = st.unpack(format, binaryvalue)
	value = value[0]

	qualityaux=False
	if qualityimage is not None:
		# read quality image
		qdataset,qrows,qcols,qbands = readtif(qualityimage)

		qband=qdataset.GetRasterBand(1)
		qbinaryvalue=qband.ReadRaster(xcoord,ycoord,1,1,buf_type=data_type)
		qvalue = st.unpack(format, qbinaryvalue)
		qvalue = qvalue[0]
		qualityaux = (qvalue==badqualityvalues).any()
	
	if (value == fillvalue) | qualityaux :
		value = np.nan

	return value

def imagestacktable(shapefile,data_frame1,data_frame2,\
					date_variable1="yeardate",date_variable2="yeardate",\
					positions1=None,positions2=None,days=None,years=1,\
					variable=2,quality_variable=7,fillvalue=None):
	'''
	some fucking info

	'''
	if days is not None:
		
		print("creating coordinates")
		# coordinates of each object in shapefile
		shpcoordinates = shapecoordinates(shapefile)

		print("merging dataframes")
		# merge coordinates to training dataframe
		merged = mergedataframes(shpcoordinates,data_frame1)

		print("searching dataframes (dates)")
		data_frames, data_length = searchdates(merged,data_frame2,\
			date_variable1=date_variable1,date_variable2=date_variable2,\
			positions1=positions1,positions2=positions2,\
			days=days)

		# output table dimensions
		minimum_length = min(data_length)
		ndata_frames = len(data_frames)
		output = np.zeros((ndata_frames,minimum_length+1))

		print("processing dataframes")
		for i in xrange(ndata_frames):


			data_frame = data_frames[i]

			output[i,0] = merged.iloc[i,0]

			for j in xrange(minimum_length):


				imagepath=data_frame.iloc[j,variable]
				qualityimagepath=data_frame.iloc[j,quality_variable]

				value = extract(merged.iloc[i,1],merged.iloc[i,2],\
				image=imagepath,\
				qualityimage=qualityimagepath,\
				data_type=16,fillvalue=fillvalue)
				output[i,j+1]=value

		return output

	else:

		print("creating coordinates")
		# coordinates of each object in shapefile
		shpcoordinates = shapecoordinates(shapefile)

		print("merging dataframes")
		# merge coordinates to training dataframe
		merged = mergedataframes(shpcoordinates,data_frame1)

		data_frames, data_length = searchdates(merged,data_frame2,\
			date_variable1=date_variable1,date_variable2=date_variable2,\
			positions1=positions1,positions2=positions2,\
			days=None,years=years)

		# output table dimensions
		minimum_length = min(data_length)
		ndata_frames = len(data_frames)
		output = np.zeros((ndata_frames,minimum_length+1))

		print("processing data frames")
		for i in xrange(ndata_frames):


			data_frame = data_frames[i]

			output[i,0] = merged.iloc[i,0]

			for j in xrange(minimum_length):


				imagepath=data_frame.iloc[j,variable]
				qualityimagepath=data_frame.iloc[j,quality_variable]

				value = extract(merged.iloc[i,1],merged.iloc[i,2],\
				image=imagepath,\
				qualityimage=qualityimagepath,\
				data_type=16,fillvalue=fillvalue)
				output[i,j+1]=value

		return output

def numpytodataframe(numpyarray,axis=0,names=True,variableprefix="v"):

	'''
	some fucking info

	'''
	
	variableprefix = str(variableprefix)

	nrows = np.shape(numpyarray)[0]
	ncols = np.shape(numpyarray)[1]

	rindex = range(nrows)
	cindex = range(ncols)
	
	if axis == 0:

		del cindex[-1]
		colnames = []
		for i in cindex:
			colnames.append(variableprefix+str(i))

		dataframe = pd.DataFrame.from_records(data = numpyarray[:,1:],index=numpyarray[:,0],columns=colnames)
	
	return dataframe

def countnas(table,axis=1):
	
	'''
	dataframe or numpy array 

	'''
	if isinstance(table,(np.ndarray, np.generic)):
		nans = np.zeros((np.shape(table)[0]))
		for i in xrange(np.shape(table)[0]):
			nans[i]=np.sum(np.isnan(table[i,:]))
	else:
		nonnans = table.count(axis=axis,numeric_only=True)
		nans = len(table.columns)-nonnans
	return(nans)

def stacknas(path,axis=1):

	'''
	some fucking info

	'''

	series_list = []
	variable_names = []
	for file in os.listdir(path):
	
		if file.endswith(".csv"):
			body = str(file).split(".",2)
			data = np.genfromtxt(path+str(file), delimiter=',')
			data = numpytodataframe(data)
			counted = countnas(data,axis=axis)
			variable_names.append(body[0])
			series_list.append(counted)
	
	concatenated = pd.concat(series_list,axis=1)
	concatenated.columns = variable_names
	return concatenated

#def removebadts(train_df,covariate_df):

def simple_features(modispath,trainingpath,outpath,axis=1):

	'''
	some fucking info

	'''

	for file in os.listdir(path):
	
		if file.endswith(".csv"):
			body = str(file).split(".",2)
			data = np.genfromtxt(path+str(file), delimiter=',')
			mask = np.isnan(data)
			table = np.ma.masked_invalid(data)

			varname1 = outpath+str(body[0])+"_mean.csv"
			variable = np.ma.mean(table,axis=axis)
			np.savetxt(varname,variable, delimiter=",")

def sliding_features(imagesdf,years=np.array([2004,2005,2006,2007,2008,2009,\
						2010,2011,2012,2013]),yeardate_variable="yeardate",\
						monthdata_variable="monthdate",variable=5,\
						quality_variable=7,badqualityvalues=[-1,2,3,255],\
						window=1,fillvalue=-3000):

	# column names in searched_data_frame
	colnames = list(imagesdf.columns.values)
	
	# finde index of date variable
	idx_names = colnames.index(yeardate_variable)

	for i in xrange(np.shape(years)[0]):
		print(i)
	#for i in xrange(1):

		base_date = float(years[i])
		subset = imagesdf.loc\
		[\
		  (imagesdf.iloc[:,idx_names] == (base_date - window))\
		| (imagesdf.iloc[:,idx_names] == (base_date))\
		| (imagesdf.iloc[:,idx_names] == (base_date + window))\
		]

		# initialize
		dataset,rows,cols,bands = readtif(subset.iloc[0,5])
		image_time_series = np.ma.zeros((len(subset.index), cols * rows),dtype=np.float64)
		image_time_seriesq = np.zeros((len(subset.index), cols * rows),dtype=bool)

		for j in xrange(len(subset.index)):
			
			# read images (variable of interest and associated quality product) 
			dataset,rows,cols,bands = readtif(subset.iloc[j,variable])
			qdataset,qrows,qcols,qbands = readtif(subset.iloc[j,quality_variable])
			
			# make numpy array and flatten
			band = dataset.GetRasterBand(1)
			band = band.ReadAsArray(0, 0, cols, rows).astype(np.int16)
			band = np.ravel(band)

			qband = qdataset.GetRasterBand(1)
			qband = qband.ReadAsArray(0, 0, cols, rows).astype(np.int16)
			qband = np.ravel(qband)

			# check which pixels have bad quality
			qualityaux = np.in1d(qband,badqualityvalues)
			image_time_seriesq[j, :] = qualityaux

			band[qualityaux]=fillvalue
			masked = band == fillvalue


			image_time_series[j, :] = np.ma.array(band,mask=masked)
			
			# close qdataset
			qdataset = None

		# bad data count
		image_time_seriesq = np.sum(image_time_seriesq,axis=0)
		print(image_time_seriesq)
		

		# image metadata
		projection = dataset.GetProjection()
		transform = dataset.GetGeoTransform()
		driver = dataset.GetDriver()

		name = "D:/Julian/64_ie_maps/rasters/covariates/"+str(int(base_date))+"_badpixels.tif"
		outData = createtif(driver,rows,cols,1,name,data_type=16)
		writetif(outData,image_time_seriesq,projection,transform,order='r')

		# close dataset properly
		outData = None

		# means
		column_means = np.ma.mean(image_time_series,axis=0,dtype=np.float64)

		# image metadata
		projection = dataset.GetProjection()
		transform = dataset.GetGeoTransform()
		driver = dataset.GetDriver()

		name = "D:/Julian/64_ie_maps/rasters/covariates/"+str(int(base_date))+"_mean.tif"
		outData = createtif(driver,rows,cols,1,name)
		writetif(outData,column_means,projection,transform,order='r')

		# close dataset properly
		outData = None

		# standard deviations
		column_standarddeviations = np.ma.std(image_time_series,axis=0,dtype=np.float64)

		# image metadata
		projection = dataset.GetProjection()
		transform = dataset.GetGeoTransform()
		driver = dataset.GetDriver()

		name = "D:/Julian/64_ie_maps/rasters/covariates/"+str(int(base_date))+"_std.tif"
		outData = createtif(driver,rows,cols,1,name)
		writetif(outData,column_standarddeviations,projection,transform,order='r')

		# close dataset properly
		outData = None

		# coefficient of variations
		column_means = 1/column_means
		coefficients_of_variation = np.multiply(column_standarddeviations,column_means)

		# image metadata
		projection = dataset.GetProjection()
		transform = dataset.GetGeoTransform()
		driver = dataset.GetDriver()

		name = "D:/Julian/64_ie_maps/rasters/covariates/"+str(int(base_date))+"_cvar.tif"
		outData = createtif(driver,rows,cols,1,name)
		writetif(outData,coefficients_of_variation,projection,transform,order='r')

		# close dataset properly
		outData = None

	# close dataset 
	dataset = None

	return True

def calculate_spectral_angle(x,y,order=1):
    '''
    This operation acts upon two dimensional numpy arrays
    
    '''
    spectral_angle_top = np.sum(np.multiply(x,y),axis=order)
    spectral_angle_bottom = 1/np.multiply(np.linalg.norm(x,axis=order),np.linalg.norm(y,axis=order))
    spectral_angle = np.arccos(np.multiply(spectral_angle_top,spectral_angle_bottom))

    return(spectral_angle)

def calculate_spectral_correlation(x,y,order=1):
    '''
    This operation acts upon two dimensional numpy arrays
    
    '''
    if (order == 1):
        x_means = np.mean(x,axis=order)
        y_means = np.mean(y,axis=order)
        x = x - x_means[:,np.newaxis]
        y = y - y_means[:,np.newaxis]

    elif (order == 0):
        x_means = np.mean(x,axis=order)
        y_means = np.mean(y,axis=order)
        x = x - x_means[np.newaxis,:]
        y = y - y_means[np.newaxis,:]

    spectral_correlation_top = np.sum(np.multiply(x,y),axis=order)
    spectral_correlation_bottom = 1/np.multiply(np.linalg.norm(x,axis=order),np.linalg.norm(y,axis=order))
    spectral_correlation = np.multiply(spectral_correlation_top,spectral_correlation_bottom)
    
    return(spectral_correlation)



