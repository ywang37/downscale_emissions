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

#######################
# Start user parameters
#

joint_zhen_dir = '../data/joint_emission_from_Zhen/'

MIX_dir = '../data/MIX/'

downscale_dir = '../data/downscale/'

start_year = 2005
end_year = 2005

month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# downscale SO2
flag_SO2 = True

# downscale NO
flag_NO = True

verbose = True

#
# End user parameters
#####################

# species to be downscaled
species = []
if flag_SO2:
    species.append('SO2')
if flag_NO:
    species.append('NO')

species_dict = {'SO2': 'SO2', 'NO': 'NOx'}

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
        post_emi = read_nc(post_file, post_var_list, verbose=verbose)

        if first_area:

            # posterior latitude and longitude
            post_lat_c = post_emi['lat']
            post_lon_c = post_emi['lon']
            post_lon_c, post_lat_c = np.meshgrid(post_lon_c, post_lat_c)
            post_lat_e, post_lon_e = \
                    calculate_pixel_edge2(post_lat_c, post_lon_c)

            calc_overlap_area_record(post_lat_e, post_lon_e,
                    mix_lat_e, mix_lon_e)

















