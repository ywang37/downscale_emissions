from pylib.Get_Pixels_Corners_Custom import *

# libraries for multiprocessing
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
from functools import partial

import numpy as np


#----------------------------------------------------------------------------
def getCoordinate(cord, resolution):
    '''
    uppLat: [float], upper latitude of the region of interest
    lowLat: [float], lower latitude of the region of interest   
    lefLon: [float], left longitude of the region of interest
    rigLon: [float], right longitude of the region of interest
    resolution: [float], resolution of the regular mesh grid
	
	Function to get the latitude and longitude of the regular mesh grid
    '''
    import numpy as np
    uppLat, lowLat, lefLon, rigLon = cord
    
    lenLat = np.int16((uppLat-lowLat)/resolution)+1
    lenLon = np.int16((rigLon-lefLon)/resolution)+1
                             
    latWeighted = (np.linspace(lowLat,uppLat,lenLat)* np.ones((lenLon,1),\
                               np.float32)).T
    lonWeighted = np.ones((lenLat,1),np.float32) * np.linspace(lefLon,rigLon,\
                         lenLon)
    
    return [lonWeighted, latWeighted]
    
#----------------------------------------------------------------------------
def split_array(len_array, processes_n):
	import numpy as np
	list_limits = []
	for ii in np.arange(0, processes_n):
		total_items = len(len_array)
		q = int(total_items/processes_n)
		r = total_items % processes_n
		limits = []
	
		for iter in np.arange(0, processes_n-1):
			if (iter <= r-1) :
				limits.append(q+1)
			else: 
				limits.append(q)
		limits.append(total_items-sum(limits))
		lio = 0
		ix=ii
	
		while ( ix >= 1 ):
			lio = lio + limits[ix-1]
			ix = ix-1
		ls = lio + limits[ii] 
		lis = len_array[lio:ls]  
		list_limits.append([lio, ls])
	return list_limits

#----------------------------------------------------------------------------
def mapRegridFun0(obsReg, cordRegCorn, winLen, cordCorn, obs, interest_pos):
#	print (interest_pos, 'here')
	
	rows = interest_pos[1] - interest_pos[0]
	new_grid   = np.zeros((rows, obsReg.shape[1]))
# 	print (new_grid.shape, 'shape')
	
	ii = 0
	result = dict(x = [], y = [], value = [])
	for i in range(interest_pos[0], interest_pos[1]):
		
		for j in range(np.shape(obsReg)[1]):
			#print ('j', j)
			cPixel = {}
			for key in cordRegCorn.keys():
				cPixel[key] = cordRegCorn[key][i,j]
	
			new_grid[ii, j] = regridPixel(cPixel, winLen, cordCorn, obs)

		ii = ii + 1


	 
	return {'sub_array': new_grid, 'rows': interest_pos}

#----------------------------------------------------------------------------
def mapRegridFun(obsReg, cordRegCorn, winLen, cordCorn, obs, keyPara, interest_pos):


	rows = interest_pos[1] - interest_pos[0]

	ii = 0
	result = dict(x = [], y = [], value = {}, cArea = [], area = [])
	for obs_key in obs.keys():
		result['value'][obs_key] = []
		
	for i in range(interest_pos[0], interest_pos[1]):

		for j in range(np.shape(obsReg)[1]): 

			cPixel = {}
			for key in cordRegCorn.keys():
				cPixel[key] = cordRegCorn[key][i,j]

			result['x'].append(i)
			result['y'].append(j)

			dataWeighted, area, cArea = regridPixel(cPixel, winLen, cordCorn, obs, keyPara)
			result['cArea'].append(cArea)
			result['area'].append(area)
			for obs_key in obs.keys():
				result['value'][obs_key].append(dataWeighted[obs_key])
		ii = ii + 1

	return result

#----------------------------------------------------------------------------
def regridMap0(cord, data, para, resolution):
	'''
    cord, boundary of the DOI
		uppLat: [float], upper latitude of the region of interest
		lowLat: [float], lower latitude of the region of interest   
		lefLon: [float], left longitude of the region of interest
		rigLon: [float], right longitude of the region of interest         
    data, [dictionary-like], latitude, longitude, and other datasets need to be
    	  resampled.
    para: [string], name (key) of the dataset to be narrowed
	resolution, [float], resolution of the regular mesh grid
	
	Update:
	MZ, 23/12/2019, project data at the begin of the workflow...	
	'''
	import numpy as np
	
	# create the standard mesh grid...
	uppLat, lowLat, lefLon, rigLon = cord
	
	narrowVar	= [para, 'latitude', 'longitude']
	narrowCord 	= [uppLat + 2*resolution, lowLat - 2*resolution, 
	               lefLon - 2*resolution, rigLon + 2*resolution] 
	           
	narrowData 	= narrowDomain(data, narrowCord, narrowVar)
	if narrowData == 0:
		regridFlag = 0
		return regridFlag, 0, 0, 0
	
	lat = narrowData['latitude']
	lon = narrowData['longitude']
	obs = narrowData[para]
			
	lonReg, latReg = getCoordinate(cord, resolution)                       
	obsReg = np.full_like(latReg, np.nan)
		
	# project the standard mesh grid on the alber equal area coordinates system
	
	xReg, yReg = alber_equal_area(lonReg, latReg)
	cordRegCorn = Calculate_Corners(xReg, yReg, tile = 'temp')
	cordRegCorn['central_lon']  = xReg
	cordRegCorn['central_lat']  = yReg
	
	cordRegCorn['raw_lat']  = latReg
	cordRegCorn['raw_lon']  = lonReg
	

	xWinLen = np.abs(xReg[0,0] - xReg[0,1])
	yWinLen = np.abs(yReg[0,0] - yReg[1,0])
	
	winFactor = 2
	
	if xWinLen > yWinLen:
		winLen = xWinLen * winFactor
	else:
		winLen = yWinLen * winFactor

	x, y = alber_equal_area(lon, lat)
	cordCorn = Calculate_Corners(x, y, tile = 'temp')
	cordCorn['central_lon']  = x
	cordCorn['central_lat']  = y
	
	cordCorn['raw_lon']  = lon
	cordCorn['raw_lat']  = lat
	
	len_array = np.arange(0,obsReg.shape[0])
	p = 32
	sub_arrays_list = split_array(len_array, p)
# 	print(sub_arrays_list)
	
	pool = Pool(processes=p)
	func = partial(mapRegridFun0, obsReg, cordRegCorn, winLen, cordCorn, obs, resolution)
	result = pool.map(func, sub_arrays_list)
	
	for item in result:
		#print(item)
		li = item['rows'][0]
		ls = item['rows'][1]
		obsReg[li:ls,:] = item['sub_array']
	
	pool.close()
	pool.join()

	regridFlag = 1	
    
	return regridFlag, lonReg.T, latReg.T, obsReg.T
	
#----------------------------------------------------------------------------
def regridMap(cord, data, para, resolution, keyPara, p = 56):
	'''
	cord, boundary of the DOI
		uppLat: [float], upper latitude of the region of interest
		lowLat: [float], lower latitude of the region of interest   
		lefLon: [float], left longitude of the region of interest
		rigLon: [float], right longitude of the region of interest         
	data, [dictionary-like], latitude, longitude, and other datasets need to be
		  resampled.
	para: [string], name (key) of the dataset to be narrowed
	resolution, [float], resolution of the regular mesh grid

	Update:
	MZ, 23/12/2019, project the data to AE at the beginning of the algorithm...	
	LC, 10/01/2020, adapt to multiprocessing...
	MZ, 20/01/2020, adapt the algorithm to multiple parameters..
	'''
	import numpy as np

# 	keyPara	= kwargs.get('keyPara', para[0])

	# create the standard mesh grid...
	uppLat, lowLat, lefLon, rigLon = cord

	narrowVar	= ['latitude', 'longitude'] + para
	narrowCord 	= [uppLat + 2*resolution, lowLat - 2*resolution, 
				   lefLon - 2*resolution, rigLon + 2*resolution] 
		   
	narrowData 	= narrowDomain(data, narrowCord, narrowVar)
	if narrowData == 0:
		regridFlag = 0
		return regridFlag, 0
		
	lat = narrowData['latitude']
	lon = narrowData['longitude']
	obs = {}
	for key_obs in para:
		obs[key_obs] = narrowData[key_obs]
		
	lonReg, latReg = getCoordinate(cord, resolution)   
	
# 	print(latReg)                    
# 	print(np.shape(lonReg))
	# project the standard mesh grid on the alber equal area coordinates system
	xReg, yReg = alber_equal_area(lonReg, latReg)
	
	cordRegCorn = Calculate_Corners(xReg, yReg, tile = 'temp')
	cordRegCorn['central_lon']  = xReg
	cordRegCorn['central_lat']  = yReg

	cordRegCorn['raw_lat']  = latReg
	cordRegCorn['raw_lon']  = lonReg
	# i may improve a little bit here
	# some lines are no need to calculate
# 	print(np.min(latReg), np.max(latReg))
# 	print(np.min(lat), np.max(lat))
# 	
# 	print('idx')
# 	print( (np.min(lat) - np.min(latReg))//resolution )
# 	print( (np.max(lat) - np.min(latReg))//resolution )
# 
# 	print( (np.min(lat) - np.max(latReg))//resolution )
# 	print( (np.max(lat) - np.max(latReg))//resolution )
	
	xWinLen = np.abs(xReg[0,0] - xReg[0,1])
	yWinLen = np.abs(yReg[0,0] - yReg[1,0])

	winFactor = 2

	if xWinLen > yWinLen:
		winLen = xWinLen * winFactor
	else:
		winLen = yWinLen * winFactor

	x, y = alber_equal_area(lon, lat)
	cordCorn = Calculate_Corners(x, y, tile = 'temp')
	cordCorn['central_lon']  = x
	cordCorn['central_lat']  = y

	cordCorn['raw_lon']  = lon
	cordCorn['raw_lat']  = lat
	
	obsReg = np.full_like(latReg, np.nan)
	
# 	print(obsReg.shape[0])
	if obsReg.shape[0] > p:
		p = p
	else:
		p = obsReg.shape[0]
	
	len_array = np.arange(0,obsReg.shape[0])
	sub_arrays_list = split_array(len_array, p)
# 	print(sub_arrays_list)

	output = {}
	pool = Pool(processes=p)
	func = partial(mapRegridFun, obsReg, cordRegCorn, winLen, cordCorn, obs, keyPara)
	result = pool.map(func, sub_arrays_list)

	# 	obsReg = np.full_like(latReg, np.nan)

	for obs_key in obs.keys():
		obsReg = np.full_like(latReg, np.nan)
		for item in result:
			obsReg[item['x'], item['y']] = item['value'][obs_key]
		output[obs_key] = obsReg
		
	cArea = np.full_like(latReg, np.nan)
	area = np.full_like(latReg, np.nan)
	
	for item in result:
		cArea[item['x'], item['y']] = item['cArea']
		area[item['x'], item['y']] = item['area']
# 		print(item['x'], item['y'], item['area'])


	pool.close()
	pool.join()

	regridFlag = 1	
	output['latitude'] = latReg
	output['longitude'] = lonReg
	output['cArea'] = cArea
	output['area'] = area
	
	return regridFlag, output
	
#----------------------------------------------------------------------------   
def regridPixel(cPixel, winLen, cordCorn, obs, keyPara):
	'''
    cPixel, [dictionary-like], contains latitude, longitude of a standard pixel  
	winLen: [float], length of the window used to be locatize the measurement
	cordCorn: [dictionary-like], pixel coordinates of the data to be resampled
	obs: [array-like], data to be resampled

	Function to regrid meaurement to one regular pixel
	''' 
	import numpy as np
	o = 1e-3
# 	print('clat: %4.3f, clon: %4.3f: '%(cPixel['raw_lat'], cPixel['raw_lon']))	
	dataWeighted = {}
	area = np.nan
	cArea = np.nan	
	surPixels, surObs = findPixel(cPixel, winLen, cordCorn, obs)

	if surPixels == 0:
# 		print('No pixel inside cell.')
		for obs_key in obs.keys():
			dataWeighted[obs_key] = np.nan
		area = np.nan
	else:
		key_cord = list(surPixels.keys())[0]
		if np.size(surPixels[key_cord]) < 1:
# 			print('Pixel amount less than 1')
			# make sure there is more than one point...
			for obs_key in obs.keys():
				dataWeighted[obs_key] = np.nan
			area = np.nan
		else:
			weights, area, cArea = getWeights(cPixel, surPixels, surObs)
			if np.nansum(area) < o:
				for obs_key in surObs:				
					dataWeighted[obs_key] = np.nan
			else:
				
				mask = surObs[keyPara]
				mask[np.where(mask != mask)] = -999
				idx = np.where( mask > 0 )
				if np.size(idx) > 0:
					weights = weights[idx]
					sumWeights = np.nansum(weights)
					if sumWeights > 0:
						weights = weights/sumWeights
						area = np.nansum(area[idx])			
						for obs_key in surObs:
							dataWeighted[obs_key] = np.nansum(weights * surObs[obs_key][idx])
					else:
						area = np.nan
						for obs_key in surObs:
							dataWeighted[obs_key] = np.nan				
				else:				
					for obs_key in obs.keys():	
						dataWeighted[obs_key] = np.nan
					area = np.nan
					cArea = np.nan

	return [dataWeighted, np.sum(area), cArea]

#----------------------------------------------------------------------------
def findPixel(cPixel, winLen, cordCorn, obs):
	'''
	clon: [float], longitude of pixel of interest
	clat: [float], latitude of pixel of interest    
	dis: [float], length of the window used to be locatize the measurement
	lon: [array-like], longitude of the region that covered by the
		 pixel of interest
	lat: [array-like], latitude of the region that covered by the
		 pixel of interest
	measure: [array-like], data to be localized

	Function to localized a small region for regriding
	'''
	import numpy as np
	clat = cPixel['central_lat']
	clon = cPixel['central_lon']

	lat = cordCorn['central_lat']
	lon = cordCorn['central_lon']

	uppLat = clat + winLen ; lowLat = clat - winLen
	lefLon = clon - winLen ; rigLon = clon + winLen

	#     print(uppLat, clat, lowLat)
	#     print(lefLon, clon, rigLon)

	result = np.where((lat >= lowLat ) & (lat <= uppLat ) & \
					  (lon >= lefLon ) & (lon <= rigLon ))

	if len(result[0]) == 0:
		return 0, 0
	H_min = np.min(result[0])
	H_max = np.max(result[0])
	V_min = np.min(result[1])
	V_max = np.max(result[1])

	surPixels = {}
	surObs = {}
	
	for key in cordCorn.keys():
		surPixels[key] = cordCorn[key][H_min:(H_max+1),V_min:(V_max+1)]
		
	for key_obs in obs.keys():
		surObs[key_obs] = obs[key_obs][H_min:(H_max+1),V_min:(V_max+1)]
    
	return surPixels, surObs

#----------------------------------------------------------------------------
def getWeights(cPixel, surPixels, obs):
	'''
	cPixel,    [dictionary-like], contains latitude, longitude of a standard 
			   pixel  
	surPixels, [dictionary-like], contains latitude, longitude of a raw pixel 
			   surround the center pixel
	obs,       [array-like], data to be resampled

	Albers Equal Area Projection is used in this function to calculate
	the overlap aera of the pixel of interest and the area

	'''
	import numpy as np
	from shapely.geometry import Polygon
	
	o = 1e-3
	cCord = [
			 (cPixel['Upper_Left_Lon'], cPixel['Upper_Left_Lat']),\
			 (cPixel['Upper_Right_Lon'], cPixel['Upper_Right_Lat']),\
		     (cPixel['Lower_Right_Lon'], cPixel['Lower_Right_Lat']),\
		     (cPixel['Lower_Left_Lon'], cPixel['Lower_Left_Lat'])
		    ]	
	cPolygon = Polygon(cCord)
	
# 	print('  - total area: %8.3f: '%(cPolygon.area))
	
	area = np.full_like(surPixels['Upper_Left_Lon'], np.nan)

	surPolygons = []
	
	Upper_Left_Lon	= surPixels['Upper_Left_Lon']
	Upper_Left_Lat	= surPixels['Upper_Left_Lat']
	Upper_Right_Lon = surPixels['Upper_Right_Lon']
	Upper_Right_Lat = surPixels['Upper_Right_Lat']
	Lower_Right_Lon = surPixels['Lower_Right_Lon']
	Lower_Right_Lat = surPixels['Lower_Right_Lat']
	Lower_Left_Lon 	= surPixels['Lower_Left_Lon']
	Lower_Left_Lat 	= surPixels['Lower_Left_Lat']
	
	
	for i in range(len(area[:,0])):
		for j in range(len(area[0,:])):
			#  modified here
			Cord = [(Upper_Left_Lon[i,j], Upper_Left_Lat[i,j]),\
					(Upper_Right_Lon[i,j], Upper_Right_Lat[i,j]),\
					(Lower_Right_Lon[i,j], Lower_Right_Lat[i,j]),\
					(Lower_Left_Lon[i,j], Lower_Left_Lat[i,j])]
			surPolygon = Polygon(Cord)
	    				 
			area[i,j] = surPolygon.intersection(cPolygon).area
			surPolygons.append(surPolygon)
			
	cArea = cPolygon.area
	overlapArea = np.nansum(area)
	if overlapArea < o:
		weights = np.full_like(surPixels['Upper_Left_Lon'], np.nan)
	else:
		weights = np.true_divide(area, np.nansum(area))


	return [weights, area, cArea]

#----------------------------------------------------------------------------	
def alber_equal_area(lon, lat, lat_0 = 40, lat_1 = 20, lat_2 = 60, lon_0 = -96):
	'''
	Function to convert ... to alber equal area
	'''
	
	import numpy as np
	from numpy import cos, sin, log

	def cal_alpha(phi, e):

		def first_term(phi, e):
	
			tan_phi = sin(phi) / ( 1 - (e*sin(phi))**2 )
	
			return tan_phi

		def second_term(phi, e):

			xx = ( 1 / (2 * e) ) * log( (1 - e * sin(phi) ) / ( 1 + e * sin(phi) ) )
	
			return xx

		alpha = (1 - e**2) * ( first_term(phi, e) - second_term(phi, e) )

		return alpha
		
	pi = 180.0 / np.pi
	R = 6378137
	# flattening
	f = 1/298.257233
	# eccentricity
	e = (2*f - f**2)**0.5
	
	lamda = lon / pi
	
	phi = lat / pi

	phi_0 = lat_0 / pi
	
	phi_1 = lat_1 / pi
	
	phi_2 = lat_2 / pi
	
	lamda_0 = lon_0 / pi
	
	alpha = cal_alpha(phi, e)
	
	alpha_0 = cal_alpha(phi_0, e)
	
	alpha_1 = cal_alpha(phi_1, e)
	
	alpha_2 = cal_alpha(phi_2, e)
	
	m1 = cos(phi_1) / ( 1 - (e * sin(phi_1))**2)**0.5
	
	m2 = cos(phi_2) / ( 1 - (e * sin(phi_2))**2)**0.5
	
	n = (m1**2 - m2**2) / (alpha_2 - alpha_1)
	
	C = m1**2 + n * alpha_1
	
	theta = n * (lamda - lamda_0)

	rho = R/n * (C - n*alpha)**0.5
	
	rho_0 =  R/n * (C - n*alpha_0)**0.5

	x = rho * sin(theta)
	
	y = rho_0 - rho*cos(theta)
	
	return x, y
	
#----------------------------------------------------------------------------	
def narrowDomain(data, cord, var):
    '''
    cord, boundary of the DOI   
    lon: [array-like], longitude of the region that covered by the
         pixel of interest
    lat: [array-like], latitude of the region that covered by the
         pixel of interest
    measure: [array-like], data to be localized
    
    Function to localized a small region for regriding
    '''
    
    import numpy as np
    lat = data['latitude']
    lon = data['longitude']

    uppLat = cord[0] ; lowLat = cord[1]
    lefLon = cord[2] ; rigLon = cord[3]
    result = np.where((lat >= lowLat ) & (lat <= uppLat ) & \
                      (lon >= lefLon ) & (lon <= rigLon ))
    if len(result[0]) == 0:
        return 0
    H_min = np.min(result[0])
    H_max = np.max(result[0])
    V_min = np.min(result[1])
    V_max = np.max(result[1])
    
    for key in var:
    	data[key] = data[key][H_min:(H_max+1),V_min:(V_max+1)]
    return data	
	
	
	
	
