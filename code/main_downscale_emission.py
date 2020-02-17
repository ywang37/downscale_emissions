"""
Created on Feburary 15, 2020

@author: Yi Wang
"""

import copy
import numpy as np
import sys

from mylib.io import read_nc
from mylib.pro_satellite.pro_satellite import calculate_pixel_edge2

sys.path.append('/Users/ywang466/small_projects/downscale_emissions/code')
from downscale.downscale import calc_overlap_area_record
from downscale.downscale import downscale_emissions
from downscale.io import write_nc
from downscale.grid_utility import get_center_index

#######################
# Start user parameters
#

joint_zhen_dir = '../data/joint_emission_from_Zhen/'

MIX_dir = '../data/MIX/'

downscale_dir = '../data/downscale/'

start_year = 2005
end_year = 2012

month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

region_limit = [-0.125, 69.875, 50.125, 150.125]

# downscale SO2
flag_SO2 = True

# downscale NO
flag_NO = True

verbose = True

#
# End user parameters
#####################

month_ind_dict = \
        {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4,  'Jun': 5, \
         'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}

# species to be downscaled
species = []
if flag_SO2:
    species.append('SO2')
if flag_NO:
    species.append('NO')

species_dict = {'SO2': 'SO2', 'NO': 'NOx'}

src_format_dict = {'SO2': 'total', 'NO': 'rate'}

units = {'SO2': 'TgS/box', 'NO': 'molec/cm2'}

# MIX sectors
sector_list = ['INDUSTRY', 'POWER', 'RESIDENTIAL', 'TRANSPORT']

# read MIX data emissions
first_spec = True
mix = {}
for spec in species:

    mix_var_list = ['lat', 'lon']

    for sector in sector_list:
        mix_var_list.append(spec + '_' + sector)

    # read MIX emissions
    mix_file = MIX_dir + 'MIX_Asia_' + spec  + '.generic.025x025.nc'
    mix_data = read_nc(mix_file, mix_var_list, verbose=verbose)

    # latitude and longitude
    if first_spec:
        mix['lat_c'] = mix_data['lat']
        mix['lon_c'] = mix_data['lon']
        first_spec = False

    # total and sectoral emissions
    mix[spec] = {}
    mix[spec][spec+'_TOTAL'] = np.zeros_like(mix_data[spec+'_'+sector_list[0]])
    for sector in sector_list:
        # sectoral emissions
        mix[spec][spec+'_'+sector] = mix_data[spec+'_'+sector]
        # total emissions
        mix[spec][spec+'_TOTAL'] += mix[spec][spec+'_'+sector]

# MIX latitude and longitude
mix_lat_c = mix['lat_c']
mix_lon_c = mix['lon_c']
mix_lon_c, mix_lat_c = np.meshgrid(mix_lon_c, mix_lat_c)
mix_lat_e, mix_lon_e = calculate_pixel_edge2(mix_lat_c, mix_lon_c)
mix['lat_e'] = mix_lat_e
mix['lon_e'] = mix_lon_e

# sub_region index
i0 = get_center_index(mix_lat_c[:,0], region_limit[0])
i1 = get_center_index(mix_lat_c[:,0], region_limit[2])
j0 = get_center_index(mix_lon_c[0,:], region_limit[1])
j1 = get_center_index(mix_lon_c[0,:], region_limit[3])

first_area = True
for year in range(start_year, end_year+1):

    print('------ processing year {} ------'.format(year))

    for spec in species:
        print('  ' + spec + ':')

        # read posterior emissions
        if first_area:
            post_var_list = month_list + ['lat', 'lon']
        else:
            post_var_list = copy.deepcopy(month_list)
        post_file = joint_zhen_dir + \
                'joint_' + species_dict[spec]  + '_' + str(year) + '.nc'
        post_emi_data = read_nc(post_file, post_var_list, verbose=verbose)

        # calculate overlapping area
        if first_area:

            # posterior latitude and longitude
            post_lat_c = post_emi_data['lat']
            post_lon_c = post_emi_data['lon']
            post_lon_c, post_lat_c = np.meshgrid(post_lon_c, post_lat_c)
            post_lat_e, post_lon_e = \
                    calculate_pixel_edge2(post_lat_c, post_lon_c)

            src_overlap_area, tar_overlap_ind = \
                    calc_overlap_area_record(post_lat_e, post_lon_e,
                    mix_lat_e, mix_lon_e)
            first_area = False

        # output dict
        out_dict = {}
        out_dict['Latitude']    = mix_lat_c[i0:i1+1,j0:j1+1]
        out_dict['Longitude']   = mix_lon_c[i0:i1+1,j0:j1+1]
        out_dict['Latitude_e']  = mix_lat_e[i0:i1+2,j0:j1+2]
        out_dict['Longitude_e'] = mix_lon_e[i0:i1+2,j0:j1+2]

        # units_dict
        units_dict = {}

        # process monthly emissions
        for month in month_list:

            print('   ' + month)
            post_month_emi = post_emi_data[month]

            # find the cloest MIX emissions in terms of time
            if int(year) <= 2008:
                offset = 0
            elif int(year) == 2009:
                offset = 12
            elif int(year) >= 2010:
                offset = 24
            else:
                print('!!!year error!!!')
                exit()
            mix_ind = month_ind_dict[month] + offset
            print('mix_ind = {}'.format(mix_ind))
            mix_month_emi = mix[spec][spec+'_TOTAL'][mix_ind,:,:]

            # downscale emission
            src_format = src_format_dict[spec]
            out_emi, out_overlap_area = \
                    downscale_emissions(src_overlap_area, tar_overlap_ind,
                    post_month_emi, mix_month_emi, src_format)

            # save data to dict
            out_dict[month] = out_emi[i0:i1+1,j0:j1+1]

            # units_dict
            units_dict[month] = units[spec]


        # output file
        downscale_file = downscale_dir + \
                'joint_' + species_dict[spec]  + '_' + str(year) + \
                '_0.25x0.25.nc'
        write_nc(downscale_file, out_dict, units_dict=units_dict,
                verbose=verbose)

            














