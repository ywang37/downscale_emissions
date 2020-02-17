"""
Created on Feburary 17, 2020

@author: Yi Wang
"""

import matplotlib.pyplot as plt
import sys


sys.path.append('/Dedicated/jwang-data/ywang/small_projects/\
downscale_emissions/shared_code')
from de_plot import plot_emissions

#######################
# Start user parameters
#

zhen_dir = '../data/joint_emission_from_Zhen/'

downscale_dir = '../data/downscale/'

fig_dir = '../figure/'

year = '2005'

month = 'Dec'

region_limit = [-0.125, 69.875, 50.125, 150.125]

emi_name_list = ['NOx']

units_dict = {'NOx': r'[molec cm$^{-2}$ s$^{-1}$]'}

verbose = True

#
# End user parameters
#####################


for emi_name in emi_name_list:

    # zhen
    plot_emissions(zhen_dir, year, month, emi_name, 'zhen',
            vmax=4e12,
            units=units_dict[emi_name])
    figname = fig_dir + 'Zhen_' + emi_name + '_' + year + '_' + month + '.png'
    plt.savefig(figname, format='png', dpi=300)

    # downscale
    plot_emissions(downscale_dir, year, month, emi_name, 'downscale',
            vmax=4e12,
            units=units_dict[emi_name])
    figname = fig_dir + 'DS_0.25x0.25_' + emi_name + \
            '_' + year + '_' + month + '.png'
    plt.savefig(figname, format='png', dpi=300)


