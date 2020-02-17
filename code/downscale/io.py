"""
Created on January 2, 2020

@author: Yi Wang
"""

from netCDF4 import Dataset
import numpy as np


def write_nc(filename, data_dict, units_dict=None, verbose=True):
    """ 
    """

    coord_name_list = [
            'Latitude',
            'Latitude_e',
            'Longitude',
            'Longitude_e'
            ]

    if verbose:
        print(' - save_ave: output ' + filename)

    # open file
    nc_f = Dataset(filename, 'w')

    # grid, _e means edge
    Latitude    = data_dict['Latitude']
    Longitude   = data_dict['Longitude']
    Latitude_e  = data_dict['Latitude_e']
    Longitude_e = data_dict['Longitude_e']

    # Dimensions of a netCDF file
    dim_lat = nc_f.createDimension('Latitude',  Latitude.shape[0])
    dim_lon = nc_f.createDimension('Longitude', Latitude.shape[1])
    dim_lat_e = nc_f.createDimension('Latitude_e',  Latitude_e.shape[0])
    dim_lon_e = nc_f.createDimension('Longitude_e', Latitude_e.shape[1])

    # create variables in a netCDF file

    # lat and lon
    Latitude_v = nc_f.createVariable('Latitude', 'f4', 
            ('Latitude', 'Longitude'))
    Longitude_v = nc_f.createVariable('Longitude', 'f4',
            ('Latitude', 'Longitude'))
    Latitude_e_v = nc_f.createVariable('Latitude_e', 'f4',
            ('Latitude_e', 'Longitude_e'))
    Longitude_e_v = nc_f.createVariable('Longitude_e', 'f4',
            ('Latitude_e', 'Longitude_e'))

    # variables
    nc_var_dict = {}
    for varname in data_dict:
        if not (varname in coord_name_list):
            nc_var = nc_f.createVariable(varname, 'f4',
                    ('Latitude', 'Longitude'))
            nc_var_dict[varname] = nc_var

    # write variables

    # lat and lon
    Latitude_v[:]    = Latitude
    Longitude_v[:]   = Longitude
    Latitude_e_v[:]  = Latitude_e
    Longitude_e_v[:] = Longitude_e

    for varname in nc_var_dict:
        nc_var_dict[varname][:] = data_dict[varname]

    # add units
    if units_dict is not None:
        for varname in units_dict:
            nc_var_dict[varname].units = units_dict[varname]

    # close file
    nc_f.close()

