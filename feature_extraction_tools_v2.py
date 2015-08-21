
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

def generatedate(data_frame,positions=None,date_variable="date",\
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
	monthdates = np.ones((n))

	for i in positions:
			if format == "AAAA*MM*DD":
				date_string = data_frame.loc[i,str(date_variable)]
				year = date_string[0:4]
				month = date_string[5:7]
	
			elif format == "DD*MM*AAAA":
				date_string = data_frame.loc[i,str(date_variable)]
				year = date_string[6:10]
				month = date_string[3:5]
			yeardates[i]=float(year)
			monthdates[i]=float(month)

	data_frame = data_frame.iloc[positions,:]
	yeardates = yeardates[positions]
	monthdates = monthdates[positions]

	data_frame['yeardate'] = pd.Series(yeardates, index=data_frame.index)
	data_frame['monthdate'] = pd.Series(monthdates, index=data_frame.index)
	
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

def extract(xcoord,ycoord,image,nodatavalue=None,data_type=32):

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

	if value == nodatavalue:
		value = np.nan

	return value


def qextract(xcoord,ycoord,image,qualityimage=None,fillvalue=-3000,\
					badqualityvalues=np.array([-1,2,3]),data_type=32):

	'''
	extract raster-value at a given location taking into account modis
	quality flags	

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

def noshpimagestacktable(data_frame1,
						 data_frame2,
						  date_variable="yeardate",
						  year=2004,
						  month=1,
						  xcoord="x",
						  ycoord="y"):
	'''
	data_frame1 contains points with coordinates
	data_frame2 contains rasters of which you want to do an point extraction 

	'''
		
	# column names in searched_data_frame
	colnames = list(data_frame1.columns.values)

	# find index of xcoord
	idx_x = colnames.index(xcoord)

	# find index of ycoord
	idx_y = colnames.index(ycoord)

	# unique layers of covariates
	layers = pd.unique(data_frame2.iloc[:,1])

	# initialize
	output = np.zeros((len(data_frame1.index),len(layers)),dtype=np.float64)

	yearmonth=str(year)+"_"+str(month)

	for i in xrange(len(data_frame1.index)):
	#for i in xrange(10):
		for j in xrange(len(layers)):
		#for j in xrange(2):
			
			imagepath=data_frame2.loc[j,yearmonth]
			value = extract(data_frame1.iloc[i,idx_x],data_frame1.iloc[i,idx_y],\
			image=imagepath,\
			data_type=32)
			output[i,j] = value

	output = pd.DataFrame(output,columns=layers,index=data_frame1.index)
	output = pd.concat([data_frame1,output],axis=1)
	return yearmonth,output

def simpleimagestacktable(shapefile,
						 data_frame1,
						 data_frame2,
						  date_variable1="yeardate",
						  idvariable="IdConglome"):
	'''
	some fucking info

	'''
		
	print("creating coordinates")
	# coordinates of each object in shapefile
	shpcoordinates = shapecoordinates(shapefile)

	print("merging dataframes")
	# merge coordinates to training dataframe
	merged = mergedataframes(shpcoordinates,data_frame1,on=idvariable)

	# column names in searched_data_frame
	colnames = list(merged.columns.values)

	# finde index of date variable
	idx_year = colnames.index(date_variable1)


	# unique layers of covariates
	layers = pd.unique(data_frame2.iloc[:,1])

	# initialize
	output = np.zeros((len(merged.index),len(layers)),dtype=np.float64)

	#for i in xrange(len(merged.index)):
	for i in xrange(1):	
		year = int(merged.iloc[i,idx_year])
		images = data_frame2[[str(year)]]
		
		for j in xrange(len(images)):
			
			imagepath=data_frame2.loc[j,str(year)]

	  		value = extract(merged.iloc[i,1],merged.iloc[i,2],\
	  		image=imagepath,\
	  		data_type=32)

	  		output[i,j] = value

	output = pd.DataFrame(output,columns=layers,index=merged.index)
	output = pd.concat([merged,output],axis=1)
	return output			

def compleximagestacktable(shapefile,data_frame1,data_frame2,\
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

def sliding_features(imagesdf,years=np.array([2010]),yeardate_variable="yeardate",\
						monthdate_variable="monthdate",variable=5,\
						quality_variable=3,badqualityvalues=[-1,2,3,255],\
						window=0,fillvalue=0,path=None):

	# column names in searched_data_frame
	colnames = list(imagesdf.columns.values)

	# variable name
	varname=colnames[variable]
	
	# finde index of date variable
	idx_year = colnames.index(yeardate_variable)
	idx_month = colnames.index(monthdate_variable)

	for i in xrange(np.shape(years)[0]):
		print(i)
	#for i in xrange(1):

		base_date = float(years[i])
		subset = imagesdf.loc\
		[\
		  (imagesdf.iloc[:,idx_year] == (base_date - window))\
		| (imagesdf.iloc[:,idx_year] == (base_date))\
		| (imagesdf.iloc[:,idx_year] == (base_date + window))\
		]

		# initialize
		dataset,rows,cols,bands = readtif(subset.iloc[0,5])
		image_time_series = np.ma.zeros((len(subset.index), cols * rows),dtype=np.float64)
		if quality_variable is not None:
			image_time_seriesq = np.zeros((len(subset.index), cols * rows),dtype=bool)

		for j in xrange(len(subset.index)):
			print(subset.iloc[j,variable])
			
			# read images (variable of interest and associated quality product) 
			dataset,rows,cols,bands = readtif(subset.iloc[j,variable])
			
			# make numpy array and flatten
			band = dataset.GetRasterBand(1)
			band = band.ReadAsArray(0, 0, cols, rows).astype(np.int16)
			band = np.ravel(band)

			if quality_variable is not None:
				qdataset,qrows,qcols,qbands = readtif(subset.iloc[j,quality_variable])	
				qband = qdataset.GetRasterBand(1)
				qband = qband.ReadAsArray(0, 0, cols, rows).astype(np.int16)
				qband = np.ravel(qband)

				# check which pixels have bad quality
				#qualityaux = np.in1d(qband,badqualityvalues)
				#qualityaux = qband > 21 
				qualityaux = qband >= 128
				image_time_seriesq[j, :] = qualityaux

				band[qualityaux]=fillvalue
			
			masked = band >= fillvalue

			image_time_series[j, :] = np.ma.array(band,mask=masked)
			
			# close qdataset
			qdataset = None

		month_base_date=1

		# bad data count
		image_time_seriesq = np.sum(image_time_seriesq,axis=0).astype(np.float32)
		save_file(dataset, image_time_seriesq, rows, cols, path, base_date,month_base_date, varname, "badpixels")

		# means
		column_means = np.ma.mean(image_time_series,axis=0,dtype=np.float64)
		save_file(dataset, column_means, rows, cols, path, base_date,month_base_date, varname, "mean")

		# # standard deviations
		column_standarddeviations = np.ma.std(image_time_series,axis=0,dtype=np.float64)
		save_file(dataset, column_standarddeviations, rows, cols, path, base_date,month_base_date, varname, "std")

		# # coefficient of variations
		# column_means = 1/column_means
		# coefficients_of_variation = np.multiply(column_standarddeviations,column_means)
		# save_file(dataset, coefficients_of_variation, rows, cols, path, base_date, varname, "cvar")

		# # medians
		# column_medians = np.ma.median(image_time_series,axis=0)
		# save_file(dataset, column_medians, rows, cols, path, base_date, varname, "median")

		# # dry season
		# subsubset= subset.loc\
		# [\
		#   (imagesdf.iloc[:,idx_month] == 1)\
		# | (imagesdf.iloc[:,idx_month] == 2)\
		# | (imagesdf.iloc[:,idx_month] == 3)\
		# | (imagesdf.iloc[:,idx_month] == 4)\
		# | (imagesdf.iloc[:,idx_month] == 12)\
		# ]

		# # initialize
		# pimage_time_series = np.ma.zeros((len(subsubset.index), cols * rows),dtype=np.float64)
		# if quality_variable is not None:
		# 	pimage_time_seriesq = np.zeros((len(subsubset.index), cols * rows),dtype=bool)

		# for j in xrange(len(subsubset.index)):

		# 	# read images (variable of interest and associated quality product) 
		# 	dataset,rows,cols,bands = readtif(subsubset.iloc[j,variable])
			
		# 	# make numpy array and flatten
		# 	band = dataset.GetRasterBand(1)
		# 	band = band.ReadAsArray(0, 0, cols, rows).astype(np.int16)
		# 	band = np.ravel(band)

		# 	if quality_variable is not None:
		# 		qdataset,qrows,qcols,qbands = readtif(subsubset.iloc[j,quality_variable])
			
		# 		qband = qdataset.GetRasterBand(1)
		# 		qband = qband.ReadAsArray(0, 0, cols, rows).astype(np.int16)
		# 		qband = np.ravel(qband)

		# 		qualityaux = np.in1d(qband,badqualityvalues)
		# 		#qualityaux = qband > 21 
		# 		pimage_time_seriesq[j, :] = qualityaux

		# 		band[qualityaux]=fillvalue
			
		# 	masked = band == fillvalue


		# 	pimage_time_series[j, :] = np.ma.array(band,mask=masked)
			
		# 	# close qdataset
		# 	qdataset = None
				
		# # means
		# column_means = np.ma.mean(pimage_time_series,axis=0,dtype=np.float64)
		# save_file(dataset, column_means, rows, cols, path, base_date, varname, "drymean")

		# # standard deviations
		# column_standarddeviations = np.ma.std(pimage_time_series,axis=0,dtype=np.float64)
		# save_file(dataset, column_standarddeviations, rows, cols, path, base_date, varname, "drystd")

		# # coefficient of variations
		# column_means = 1/column_means
		# coefficients_of_variation = np.multiply(column_standarddeviations,column_means)
		# save_file(dataset, coefficients_of_variation, rows, cols, path, base_date, varname, "drycvar")

		# # medians
		# column_medians = np.ma.median(pimage_time_series,axis=0)
		# save_file(dataset, column_medians, rows, cols, path, base_date, varname, "drymedian")

		# # wet season
		# subsubset= subset.loc\
		# [\
		#   (imagesdf.iloc[:,idx_month] == 5)\
		# | (imagesdf.iloc[:,idx_month] == 6)\
		# | (imagesdf.iloc[:,idx_month] == 7)\
		# | (imagesdf.iloc[:,idx_month] == 8)\
		# | (imagesdf.iloc[:,idx_month] == 9)\
		# | (imagesdf.iloc[:,idx_month] == 10)\
		# | (imagesdf.iloc[:,idx_month] == 11)\
		# ]

		# # initialize
		# pimage_time_series = np.ma.zeros((len(subsubset.index), cols * rows),dtype=np.float64)

		# if quality_variable is not None:
		# 	pimage_time_seriesq = np.zeros((len(subsubset.index), cols * rows),dtype=bool)

		# for j in xrange(len(subsubset.index)):
			
		# 	# read images (variable of interest and associated quality product) 
		# 	dataset,rows,cols,bands = readtif(subsubset.iloc[j,variable])
			
			
		# 	# make numpy array and flatten
		# 	band = dataset.GetRasterBand(1)
		# 	band = band.ReadAsArray(0, 0, cols, rows).astype(np.int16)
		# 	band = np.ravel(band)

		# 	if quality_variable is not None:
		# 		qdataset,qrows,qcols,qbands = readtif(subsubset.iloc[j,quality_variable])

		# 		qband = qdataset.GetRasterBand(1)
		# 		qband = qband.ReadAsArray(0, 0, cols, rows).astype(np.int16)
		# 		qband = np.ravel(qband)

		# 		qualityaux = np.in1d(qband,badqualityvalues)
		# 		#qualityaux = qband > 21 
		# 		pimage_time_seriesq[j, :] = qualityaux

		# 		band[qualityaux]=fillvalue
			
		# 	masked = band == fillvalue

		# 	pimage_time_series[j, :] = np.ma.array(band,mask=masked)
			
		# 	# close qdataset
		# 	qdataset = None
				
		# # means
		# column_means = np.ma.mean(pimage_time_series,axis=0,dtype=np.float64)
		# save_file(dataset, column_means, rows, cols, path, base_date, varname, "wetmean")

		# # standard deviations
		# column_standarddeviations = np.ma.std(pimage_time_series,axis=0,dtype=np.float64)
		# save_file(dataset, column_standarddeviations, rows, cols, path, base_date, varname, "wetstd")
		
		# # coefficient of variations
		# column_means = 1/column_means
		# coefficients_of_variation = np.multiply(column_standarddeviations,column_means)
		# save_file(dataset, coefficients_of_variation, rows, cols, path, base_date, varname, "wetcvar")

		# # medians
		# column_medians = np.ma.median(pimage_time_series,axis=0)
		# save_file(dataset, column_medians, rows, cols, path, base_date, varname, "wetmedian")

		# # monthly features
		# for m in range(1,13):
		# 	subsubset= subset.loc\
		# 	[\
		#   	(imagesdf.iloc[:,idx_month] == m)\
		# 	]

		# 	# initialize
		# 	pimage_time_series = np.ma.zeros((len(subsubset.index), cols * rows),dtype=np.float64)

		# 	if quality_variable is not None:
		# 		pimage_time_seriesq = np.zeros((len(subsubset.index), cols * rows),dtype=bool)

		# 	for j in xrange(len(subsubset.index)):
			
		# 		# read images (variable of interest and associated quality product) 
		# 		dataset,rows,cols,bands = readtif(subsubset.iloc[j,variable])
				
			
		# 		# make numpy array and flatten
		# 		band = dataset.GetRasterBand(1)
		# 		band = band.ReadAsArray(0, 0, cols, rows).astype(np.int16)
		# 		band = np.ravel(band)

		# 		if quality_variable is not None:
		# 			qdataset,qrows,qcols,qbands = readtif(subsubset.iloc[j,quality_variable])
				
		# 			qband = qdataset.GetRasterBand(1)
		# 			qband = qband.ReadAsArray(0, 0, cols, rows).astype(np.int16)
		# 			qband = np.ravel(qband)

		# 			qualityaux = np.in1d(qband,badqualityvalues)
		# 			#qualityaux = qband > 21 
		# 			pimage_time_seriesq[j, :] = qualityaux

		# 			band[qualityaux]=fillvalue
				
		# 		masked = band == fillvalue

		# 		pimage_time_series[j, :] = np.ma.array(band,mask=masked)
			
		# 	# close qdataset
		# 	qdataset = None

		# 	# means
		# 	#column_means = np.ma.mean(pimage_time_series,axis=0,dtype=np.float64)

		# 	# image metadata
		# 	#projection = dataset.GetProjection()
		# 	#transform = dataset.GetGeoTransform()
		# 	#driver = dataset.GetDriver()

		# 	#name = path+str(int(base_date))+"/"+str(int(base_date))+"month"+str(m)+"_"+varname+"_mean.tif"
		# 	#outData = createtif(driver,rows,cols,1,name)
		# 	#writetif(outData,column_means,projection,transform,order='r')

		# 	# close dataset properly
		# 	#outData = None

		# 	# medians
		# 	column_medians = np.ma.median(pimage_time_series,axis=0)
		# 	save_file(dataset, column_medians, rows, cols, path, base_date, varname, "median")
			
		# # Percentiles
		# image_time_series = np.ma.filled(image_time_series,fill_value=np.nan)

		# # 20%
		# p20 = np.nanpercentile(image_time_series,20,axis=0)
		# p20[np.isnan(p20)]=0
		# save_file(dataset, p20, rows, cols, path, base_date, varname, "perc20")

		# # 30%
		# p30 = np.nanpercentile(image_time_series,30,axis=0)
		# p30[np.isnan(p30)]=0
		# save_file(dataset, p90, rows, cols, path, base_date, varname, "perc30")

		# # 70%
		# p70 = np.nanpercentile(image_time_series,70,axis=0)
		# p70[np.isnan(p70)]=0
		# save_file(dataset, p70, rows, cols, path, base_date, varname, "perc70")

		# # 90%
		# p90 = np.nanpercentile(image_time_series,90,axis=0)
		# p90[np.isnan(p90)]=0
		# save_file(dataset, p90, rows, cols, path, base_date, varname, "perc90")

	# close dataset 
	dataset = None

	return True

def sliding_features_months(imagesdf,\
							years=np.array([2010]),\
							months=np.array([1]),\
							yeardate_variable="yeardate",\
						monthdate_variable="monthdate",\
						variable=5,\
						quality_variable=7,\
						badqualityvalues=[-1,2,3,255],\
						window=1,fillvalue=0,path=None):

# years years=np.array([2004,2005,2006,2007,2008,2009,\
#						2010,2011,2012,2013])

	# column names in searched_data_frame
	colnames = list(imagesdf.columns.values)

	# variable name
	varname=colnames[variable]
	print(varname)
	
	# finde index of date variable
	idx_year = colnames.index(yeardate_variable)
	idx_month = colnames.index(monthdate_variable)

	for i in xrange(np.shape(years)[0]):
		print(i)
		for j in xrange(np.shape(months)[0]):
			print(j)

			base_date = float(years[i])
			month_base_date = float(months[j])

			if j==0:
				subset = imagesdf.loc\
				[\
			  	(imagesdf.iloc[:,idx_year] == (base_date - window))\
				| (imagesdf.iloc[:,idx_year] == (base_date))\
				| (imagesdf.iloc[:,idx_year] == (base_date + window))\
				]
			else:

				subset = imagesdf.loc\
				[\
			  	 (imagesdf.iloc[:,idx_year] == (base_date - window))\
				| (imagesdf.iloc[:,idx_year] == (base_date))\
				| (imagesdf.iloc[:,idx_year] == (base_date + window))\
				| (imagesdf.iloc[:,idx_year] == (base_date + window+1))\
				]

				# drop bad rows
				subset = subset.drop(subset[(subset.yeardate==(base_date-window)) & (subset.monthdate<month_base_date)].index)
				subset = subset.drop(subset[(subset.yeardate==(base_date+window+1)) & (subset.monthdate>=month_base_date)].index)

			# initialize
			dataset,rows,cols,bands = readtif(subset.iloc[0,5])
			image_time_series = np.ma.zeros((len(subset.index), cols * rows),dtype=np.float64)
			if quality_variable is not None:
				image_time_seriesq = np.zeros((len(subset.index), cols * rows),dtype=bool)

			for j in xrange(len(subset.index)):
				
				# read images (variable of interest and associated quality product) 
				dataset,rows,cols,bands = readtif(subset.iloc[j,variable])
				
				# make numpy array and flatten
				band = dataset.GetRasterBand(1)
				band = band.ReadAsArray(0, 0, cols, rows).astype(np.int16)
				band = np.ravel(band)

				if quality_variable is not None:
					qdataset,qrows,qcols,qbands = readtif(subset.iloc[j,quality_variable])	
					qband = qdataset.GetRasterBand(1)
					qband = qband.ReadAsArray(0, 0, cols, rows).astype(np.int16)
					qband = np.ravel(qband)

					# check which pixels have bad quality
					qualityaux = np.in1d(qband,badqualityvalues)
					#qualityaux = qband > 21 
					image_time_seriesq[j, :] = qualityaux

					band[qualityaux]=fillvalue
				
				masked = band == fillvalue

				image_time_series[j, :] = np.ma.array(band,mask=masked)
				
				# close qdataset
				qdataset = None

			print("first variables")

			# bad data count
			# image_time_seriesq = np.sum(image_time_seriesq,axis=0)
			# save_file(dataset, image_time_seriesq, rows, cols, path, base_date,month_base_date, varname, "badpixels")

			# means
			column_means = np.ma.mean(image_time_series,axis=0,dtype=np.float64)
			save_file(dataset, column_means, rows, cols, path, base_date,month_base_date, varname, "mean")

			# standard deviations
			column_standarddeviations = np.ma.std(image_time_series,axis=0,dtype=np.float64)
			save_file(dataset, column_standarddeviations, rows, cols, path, base_date,month_base_date, varname, "std")

			# coefficient of variations
			column_means = 1/column_means
			coefficients_of_variation = np.multiply(column_standarddeviations,column_means)
			save_file(dataset, coefficients_of_variation, rows, cols, path, base_date,month_base_date, varname, "cvar")

			# medians
			column_medians = np.ma.median(image_time_series,axis=0)
			save_file(dataset, column_medians, rows, cols, path, base_date,month_base_date, varname, "median")


			print("seasonal variables")
			# dry season
			subsubset= subset.loc\
			[\
			  (imagesdf.iloc[:,idx_month] == 1)\
			| (imagesdf.iloc[:,idx_month] == 2)\
			| (imagesdf.iloc[:,idx_month] == 3)\
			| (imagesdf.iloc[:,idx_month] == 4)\
			| (imagesdf.iloc[:,idx_month] == 12)\
			]

			# initialize
			pimage_time_series = np.ma.zeros((len(subsubset.index), cols * rows),dtype=np.float64)
			if quality_variable is not None:
				pimage_time_seriesq = np.zeros((len(subsubset.index), cols * rows),dtype=bool)

			for j in xrange(len(subsubset.index)):

				# read images (variable of interest and associated quality product) 
				dataset,rows,cols,bands = readtif(subsubset.iloc[j,variable])
				
				# make numpy array and flatten
				band = dataset.GetRasterBand(1)
				band = band.ReadAsArray(0, 0, cols, rows).astype(np.int16)
				band = np.ravel(band)

				if quality_variable is not None:
					qdataset,qrows,qcols,qbands = readtif(subsubset.iloc[j,quality_variable])
				
					qband = qdataset.GetRasterBand(1)
					qband = qband.ReadAsArray(0, 0, cols, rows).astype(np.int16)
					qband = np.ravel(qband)

					qualityaux = np.in1d(qband,badqualityvalues)
					#qualityaux = qband > 21 
					pimage_time_seriesq[j, :] = qualityaux

					band[qualityaux]=fillvalue
				
				masked = band == fillvalue


				pimage_time_series[j, :] = np.ma.array(band,mask=masked)
				
				# close qdataset
				qdataset = None
					
			# means
			column_means = np.ma.mean(pimage_time_series,axis=0,dtype=np.float64)
			save_file(dataset, column_means, rows, cols, path, base_date,month_base_date, varname, "drymean")

			# standard deviations
			column_standarddeviations = np.ma.std(pimage_time_series,axis=0,dtype=np.float64)
			save_file(dataset, column_standarddeviations, rows, cols, path, base_date,month_base_date, varname, "drystd")

			# coefficient of variations
			column_means = 1/column_means
			coefficients_of_variation = np.multiply(column_standarddeviations,column_means)
			save_file(dataset, coefficients_of_variation, rows, cols, path, base_date,month_base_date, varname, "drycvar")

			# medians
			column_medians = np.ma.median(pimage_time_series,axis=0)
			save_file(dataset, column_medians, rows, cols, path, base_date,month_base_date, varname, "drymedian")

			# wet season
			subsubset= subset.loc\
			[\
			  (imagesdf.iloc[:,idx_month] == 5)\
			| (imagesdf.iloc[:,idx_month] == 6)\
			| (imagesdf.iloc[:,idx_month] == 7)\
			| (imagesdf.iloc[:,idx_month] == 8)\
			| (imagesdf.iloc[:,idx_month] == 9)\
			| (imagesdf.iloc[:,idx_month] == 10)\
			| (imagesdf.iloc[:,idx_month] == 11)\
			]

			# initialize
			pimage_time_series = np.ma.zeros((len(subsubset.index), cols * rows),dtype=np.float64)

			if quality_variable is not None:
				pimage_time_seriesq = np.zeros((len(subsubset.index), cols * rows),dtype=bool)

			for j in xrange(len(subsubset.index)):
				
				# read images (variable of interest and associated quality product) 
				dataset,rows,cols,bands = readtif(subsubset.iloc[j,variable])
				
				
				# make numpy array and flatten
				band = dataset.GetRasterBand(1)
				band = band.ReadAsArray(0, 0, cols, rows).astype(np.int16)
				band = np.ravel(band)

				if quality_variable is not None:
					qdataset,qrows,qcols,qbands = readtif(subsubset.iloc[j,quality_variable])

					qband = qdataset.GetRasterBand(1)
					qband = qband.ReadAsArray(0, 0, cols, rows).astype(np.int16)
					qband = np.ravel(qband)

					qualityaux = np.in1d(qband,badqualityvalues)
					#qualityaux = qband > 21 
					pimage_time_seriesq[j, :] = qualityaux

					band[qualityaux]=fillvalue
				
				masked = band == fillvalue

				pimage_time_series[j, :] = np.ma.array(band,mask=masked)
				
				# close qdataset
				qdataset = None
					
			# means
			column_means = np.ma.mean(pimage_time_series,axis=0,dtype=np.float64)
			save_file(dataset, column_means, rows, cols, path, base_date,month_base_date, varname, "wetmean")

			# standard deviations
			column_standarddeviations = np.ma.std(pimage_time_series,axis=0,dtype=np.float64)
			save_file(dataset, column_standarddeviations, rows, cols, path, base_date,month_base_date, varname, "wetstd")
			
			# coefficient of variations
			column_means = 1/column_means
			coefficients_of_variation = np.multiply(column_standarddeviations,column_means)
			save_file(dataset, coefficients_of_variation, rows, cols, path, base_date,month_base_date, varname, "wetcvar")

			# medians
			column_medians = np.ma.median(pimage_time_series,axis=0)
			save_file(dataset, column_medians, rows, cols, path, base_date,month_base_date, varname, "wetmedian")

			print("percentiles")
				
			# Percentiles
			image_time_series = np.ma.filled(image_time_series,fill_value=np.nan)

			# 20%
			p20 = np.nanpercentile(image_time_series,20,axis=0)
			p20[np.isnan(p20)]=0
			save_file(dataset, p20, rows, cols, path, base_date,month_base_date, varname, "perc20")

			# 35%
			p35 = np.nanpercentile(image_time_series,35,axis=0)
			p35[np.isnan(p35)]=0
			save_file(dataset, p35, rows, cols, path, base_date,month_base_date, varname, "perc35")

			# 65%
			p65 = np.nanpercentile(image_time_series,65,axis=0)
			p65[np.isnan(p65)]=0
			save_file(dataset, p65, rows, cols, path, base_date,month_base_date, varname, "perc65")

			# 80%
			p80 = np.nanpercentile(image_time_series,80,axis=0)
			p80[np.isnan(p80)]=0
			save_file(dataset, p80, rows, cols, path, base_date,month_base_date, varname, "perc80")

			# 95%
			p95 = np.nanpercentile(image_time_series,95,axis=0)
			p95[np.isnan(p95)]=0
			save_file(dataset, p95, rows, cols, path, base_date,month_base_date, varname, "perc95")

	# close dataset 
	dataset = None

	return True

def sliding_features_months_gpp(imagesdf,\
							years=np.array([2010]),\
							months=np.array([1]),\
							yeardate_variable="yeardate",\
						monthdate_variable="monthdate",\
						variable=5,\
						quality_variable=7,\
						badqualityvalues=[-1,2,3,255],\
						window=1,fillvalue=0,path=None):

# years years=np.array([2004,2005,2006,2007,2008,2009,\
#						2010,2011,2012,2013])

	# column names in searched_data_frame
	colnames = list(imagesdf.columns.values)

	# variable name
	varname=colnames[variable]
	print(varname)
	
	# finde index of date variable
	idx_year = colnames.index(yeardate_variable)
	idx_month = colnames.index(monthdate_variable)

	for i in xrange(np.shape(years)[0]):
		print(i)
		for j in xrange(np.shape(months)[0]):
			print(j)

			base_date = float(years[i])
			month_base_date = float(months[j])

			if j==0:
				subset = imagesdf.loc\
				[\
			  	(imagesdf.iloc[:,idx_year] == (base_date - window))\
				| (imagesdf.iloc[:,idx_year] == (base_date))\
				| (imagesdf.iloc[:,idx_year] == (base_date + window))\
				]
			else:

				subset = imagesdf.loc\
				[\
			  	 (imagesdf.iloc[:,idx_year] == (base_date - window))\
				| (imagesdf.iloc[:,idx_year] == (base_date))\
				| (imagesdf.iloc[:,idx_year] == (base_date + window))\
				| (imagesdf.iloc[:,idx_year] == (base_date + window+1))\
				]

				# drop bad rows
				subset = subset.drop(subset[(subset.yeardate==(base_date-window)) & (subset.monthdate<month_base_date)].index)
				subset = subset.drop(subset[(subset.yeardate==(base_date+window+1)) & (subset.monthdate>=month_base_date)].index)

			# initialize
			dataset,rows,cols,bands = readtif(subset.iloc[0,variable])
			image_time_series = np.ma.zeros((len(subset.index), cols * rows),dtype=np.float64)
			if quality_variable is not None:
				image_time_seriesq = np.zeros((len(subset.index), cols * rows),dtype=bool)

			for j in xrange(len(subset.index)):
				
				# read images (variable of interest and associated quality product) 
				dataset,rows,cols,bands = readtif(subset.iloc[j,variable])
				
				# make numpy array and flatten
				band = dataset.GetRasterBand(1)
				band = band.ReadAsArray(0, 0, cols, rows).astype(np.int16)
				band = np.ravel(band)

				if quality_variable is not None:
					qdataset,qrows,qcols,qbands = readtif(subset.iloc[j,quality_variable])	
					qband = qdataset.GetRasterBand(1)
					qband = qband.ReadAsArray(0, 0, cols, rows).astype(np.int16)
					qband = np.ravel(qband)

					# check which pixels have bad quality
					qualityaux = np.in1d(qband,badqualityvalues)
					#qualityaux = qband > 21 
					image_time_seriesq[j, :] = qualityaux

					band[qualityaux]=fillvalue
				
				masked = band >= fillvalue

				image_time_series[j, :] = np.ma.array(band,mask=masked)
				
				# close qdataset
				qdataset = None

			print("first variables")

			# bad data count
			# image_time_seriesq = np.sum(image_time_seriesq,axis=0)
			# save_file(dataset, image_time_seriesq, rows, cols, path, base_date,month_base_date, varname, "badpixels")

			# means
			column_means = np.ma.mean(image_time_series,axis=0,dtype=np.float64)
			save_file(dataset, column_means, rows, cols, path, base_date,month_base_date, varname, "mean")

			# standard deviations
			column_standarddeviations = np.ma.std(image_time_series,axis=0,dtype=np.float64)
			save_file(dataset, column_standarddeviations, rows, cols, path, base_date,month_base_date, varname, "std")

			print("seasonal variables")
			# dry season
			subsubset= subset.loc\
			[\
			  (imagesdf.iloc[:,idx_month] == 1)\
			| (imagesdf.iloc[:,idx_month] == 2)\
			| (imagesdf.iloc[:,idx_month] == 3)\
			| (imagesdf.iloc[:,idx_month] == 4)\
			| (imagesdf.iloc[:,idx_month] == 12)\
			]

			# initialize
			pimage_time_series = np.ma.zeros((len(subsubset.index), cols * rows),dtype=np.float64)
			if quality_variable is not None:
				pimage_time_seriesq = np.zeros((len(subsubset.index), cols * rows),dtype=bool)

			for j in xrange(len(subsubset.index)):

				# read images (variable of interest and associated quality product) 
				dataset,rows,cols,bands = readtif(subsubset.iloc[j,variable])
				
				# make numpy array and flatten
				band = dataset.GetRasterBand(1)
				band = band.ReadAsArray(0, 0, cols, rows).astype(np.int16)
				band = np.ravel(band)

				if quality_variable is not None:
					qdataset,qrows,qcols,qbands = readtif(subsubset.iloc[j,quality_variable])
				
					qband = qdataset.GetRasterBand(1)
					qband = qband.ReadAsArray(0, 0, cols, rows).astype(np.int16)
					qband = np.ravel(qband)

					qualityaux = np.in1d(qband,badqualityvalues)
					#qualityaux = qband > 21 
					pimage_time_seriesq[j, :] = qualityaux

					band[qualityaux]=fillvalue
				
				masked = band >= fillvalue


				pimage_time_series[j, :] = np.ma.array(band,mask=masked)
				
				# close qdataset
				qdataset = None
					
			# means
			column_means = np.ma.mean(pimage_time_series,axis=0,dtype=np.float64)
			save_file(dataset, column_means, rows, cols, path, base_date,month_base_date, varname, "drymean")

			# standard deviations
			column_standarddeviations = np.ma.std(pimage_time_series,axis=0,dtype=np.float64)
			save_file(dataset, column_standarddeviations, rows, cols, path, base_date,month_base_date, varname, "drystd")

			# wet season
			subsubset= subset.loc\
			[\
			  (imagesdf.iloc[:,idx_month] == 5)\
			| (imagesdf.iloc[:,idx_month] == 6)\
			| (imagesdf.iloc[:,idx_month] == 7)\
			| (imagesdf.iloc[:,idx_month] == 8)\
			| (imagesdf.iloc[:,idx_month] == 9)\
			| (imagesdf.iloc[:,idx_month] == 10)\
			| (imagesdf.iloc[:,idx_month] == 11)\
			]

			# initialize
			pimage_time_series = np.ma.zeros((len(subsubset.index), cols * rows),dtype=np.float64)

			if quality_variable is not None:
				pimage_time_seriesq = np.zeros((len(subsubset.index), cols * rows),dtype=bool)

			for j in xrange(len(subsubset.index)):
				
				# read images (variable of interest and associated quality product) 
				dataset,rows,cols,bands = readtif(subsubset.iloc[j,variable])
				
				
				# make numpy array and flatten
				band = dataset.GetRasterBand(1)
				band = band.ReadAsArray(0, 0, cols, rows).astype(np.int16)
				band = np.ravel(band)

				if quality_variable is not None:
					qdataset,qrows,qcols,qbands = readtif(subsubset.iloc[j,quality_variable])

					qband = qdataset.GetRasterBand(1)
					qband = qband.ReadAsArray(0, 0, cols, rows).astype(np.int16)
					qband = np.ravel(qband)

					qualityaux = np.in1d(qband,badqualityvalues)
					#qualityaux = qband > 21 
					pimage_time_seriesq[j, :] = qualityaux

					band[qualityaux]=fillvalue
				
				masked = band >= fillvalue

				pimage_time_series[j, :] = np.ma.array(band,mask=masked)
				
				# close qdataset
				qdataset = None
					
			# means
			column_means = np.ma.mean(pimage_time_series,axis=0,dtype=np.float64)
			save_file(dataset, column_means, rows, cols, path, base_date,month_base_date, varname, "wetmean")

			# standard deviations
			column_standarddeviations = np.ma.std(pimage_time_series,axis=0,dtype=np.float64)
			save_file(dataset, column_standarddeviations, rows, cols, path, base_date,month_base_date, varname, "wetstd")

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

def save_file(dataset, data, rows, cols, path, base_date,month_base_date, varname, sufix):
	"""
	This method saves data to a tif file using the provided sufix.

	"""

	# # filter raster
	#datasetf,rows,cols,bands = readtif("D:/Julian/64_ie_maps/rasters/filter/bov_cbz_km2.tif")
	#bandf = datasetf.GetRasterBand(1)
	#bandf = bandf.ReadAsArray(0, 0, cols, rows).astype(np.float64)
	#bandf = np.ravel(bandf)

	# mexico body mask
	#baddatamask = (bandf < 0) | (data == 0) 

	# image metadata
	projection = dataset.GetProjection()
	transform = dataset.GetGeoTransform()
	driver = dataset.GetDriver()
	outpath = path + str(int(base_date)) + "/" + str(int(month_base_date)) +"/"
	if not os.path.exists(outpath):
			os.makedirs(outpath)
	name = outpath + str(int(base_date)) +"_"+str(int(month_base_date))+ "_" + varname+"_"+sufix + ".tif"
	outData = createtif(driver, rows, cols, 1, name)
	#data[baddatamask] = np.nan
	writetif(outData, data, projection, transform, order='r')

	# close dataset properly
	outData = None	
