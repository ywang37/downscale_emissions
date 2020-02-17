"""
Created on Feburary 17, 2020

@author: Yi Wang
"""

import cartopy
import cartopy.crs as ccrs
import copy
import matplotlib.pyplot as plt
import numpy as np

from mylib.borders.China.china import add_China_province
from mylib.cartopy_plot import cartopy_plot
from mylib.colormap.WhGrYlRd_map import WhGrYlRd_map
from mylib.constants import avo, mol_N
from mylib.grid_utility import grid_area_1
from mylib.io import read_nc
from mylib.layout import h_1_ax
from mylib.pro_satellite.pro_satellite import calculate_pixel_edge2

month_dict = {
        'Jan'      : 31,
        'Feb'      : 28,
        'Feb_leap' : 29,
        'Mar'      : 31,
        'Apr'      : 30,
        'May'      : 31,
        'Jun'      : 30,
        'Jul'      : 31,
        'Aug'      : 31,
        'Sep'      : 30,
        'Oct'      : 31,
        'Nov'      : 30,
        'Dec'      : 31,
        }

def plot_emissions(data_dir, year, month, emi_name, scene,
        vmin=0.0, vmax=None,
        region_limit=[-0.25, 69.75, 50.25, 150.25], units=''):
    """
    """

    # filename
    if scene == 'zhen':
        filename = data_dir + 'joint_' + emi_name + '_' + year + '.nc'
    elif scene == 'downscale':
        filename = data_dir + 'joint_' + emi_name + '_' + year + \
                '_0.25x0.25.nc'
    
    # read data
    if scene == 'zhen':
        varname_list = ['lat', 'lon', month]    
        data_dict = read_nc(filename, varname_list, verbose=True)
        lat_c = data_dict['lat']
        lon_c = data_dict['lon']
        lon_c, lat_c = np.meshgrid(lon_c, lat_c)
        lat_e, lon_e = calculate_pixel_edge2(lat_c, lon_c)
    elif scene == 'downscale':
        varname_list = ['Latitude_e', 'Longitude_e', month] 
        data_dict = read_nc(filename, varname_list, verbose=True)
        lat_e = data_dict['Latitude_e']
        lon_e = data_dict['Longitude_e']
    emissions = data_dict[month]

    xtick = np.arange(-180.0, 180.1, 20)
    ytick = np.arange(-90.0, 90.1, 10)
    title = month + ' ' + year
    pout = cartopy_plot(lon_e, lat_e, emissions, cbar=False,
            title=title,
            vmin=vmin, vmax=vmax,
            cl_res='50m',
            cmap=copy.copy(WhGrYlRd_map),
            xtick=xtick, ytick=ytick)
    ax = pout['ax']
    ax.add_feature(cartopy.feature.BORDERS)
    #add_China_province(ax)
    ax.set_xlim(region_limit[1], region_limit[3])
    ax.set_ylim(region_limit[0], region_limit[2])
    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.25, right=0.75)
    cax = h_1_ax(pout['fig'], pout['ax'])
    cbar_prop = {'orientation' : 'horizontal'}
    cb = plt.colorbar(pout['mesh'], cax=cax, orientation='horizontal')
    cb.set_label(units)

    # calculate total emissions
    if emi_name == 'NOx':
        lon_int = lon_e[0,1] - lon_e[0,0]
        nlon = lon_e.shape[1] - 1
        area_2D = grid_area_1(lat_e[:,0], lon_int, nlon)
        month_days = month_dict[month]
        total_emi = np.sum(
                emissions * 1e4 * area_2D * 3600.0 * 24.0 * month_days \
                        / avo * mol_N
                )
        total_emi = total_emi / 1e12 # g => Tg
        ax.text(128.0, 20.0, '{:.3f} Tg N'.format(total_emi), 
                transform=ccrs.PlateCarree(), color='r')
        print(total_emi)












