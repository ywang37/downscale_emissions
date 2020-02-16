"""
Created on Feburary 15, 2020

@author: Yi Wang
"""

import numpy as np

#
#------------------------------------------------------------------------------
#
def search_possible_overlap_grid(limit_src, lat_e_tar, lon_e_tar):
    """ Find all target grids that overlap with source limit region

    Parameters
    ----------
    limit_src : 1-D array-like
        Source region limit
        [lat_min, lon_min, lat_max, lon_max]
    lat_e_tar: 2-D array
        Latitude edges of 2-D target grid
    lon_e_tar: 2-D array
        Longitude edges of 2-D target grid

    Returns
    ind_dict :
        Keys:
            find: bool variabe. True and False means find
                and not find possible overlap grid, respectively.
            lat_i0: target grid latitude start index
            lat_i1: target grid latitude end index
            lon_j0: target grid longitude start index
            lon_j1: target grdi longitude end index
    """

    src_lat_min = limit_src[0]
    src_lat_max = limit_src[2]
    src_lon_min = limit_src[1]
    src_lon_max = limit_src[3]

    # possible overlap grid edges
    flag_lat = np.logical_and(lat_e_tar >= src_lat_min, 
            lat_e_tar <= src_lat_max)
    flag_lon = np.logical_and(lon_e_tar >= src_lon_min, 
            lon_e_tar <= src_lon_max)
    flag = np.logical_and(flag_lat, flag_lon)

    ind_dict = {}
    if np.sum(flag) == 0:
        ind_dict['find'] = False
        return ind_dict
    else:
        ind_dict['find'] = True

    # target dimension
    dim_tar = lat_e_tar.shape
    nlate = dim_tar[0]
    nlone = dim_tar[1]

    # find lat_i0: target grid latitude start index
    for i0 in range(nlate):
        if np.any(flag[i0,:]):
            ind_dict['lat_i0'] = i0
            if ind_dict['lat_i0'] > 0:
                ind_dict['lat_i0'] -= 1
            break

    # find lat_i1: target grid latitude end index
    for i1 in np.array(range(nlate))[::-1]:
        if np.any(flag[i1,:]):
            ind_dict['lat_i1'] = i1
            if ind_dict['lat_i1'] < (nlate-1):
                ind_dict['lat_i1'] += 1
            break

    # find lon_j0: target grid longitude start index
    for j0 in range(nlone):
        if np.any(flag[:,j0]):
            ind_dict['lon_j0'] = j0
            if ind_dict['lon_j0'] > 0:
                ind_dict['lon_j0'] -= 1
            break

    # find lon_j1: target grdi longitude end index
    for j1 in np.array(range(nlone))[::-1]:
        if np.any(flag[:,j1]):
            ind_dict['lon_j1'] = j1
            if ind_dict['lon_j1'] < (nlone-1):
                ind_dict['lon_j1'] += 1
            break

    return ind_dict
#
#------------------------------------------------------------------------------
#
def calc_overlap_area_record(lat_e_src, lon_e_src, lat_e_tar, lon_e_tar,
        flag_src=None):
    """ Calculate overlapping area of source grids and
    target grids

    Parameters
    ----------
    lat_e_src: 2-D array
        Latitude edges of 2-D source grid
    lon_e_src: 2-D array
        Longitude edges of 2-D source grid
    lat_e_tar: 2-D array
        Latitude edges of 2-D target grid
    lon_e_tar: 2-D array
        Longitude edges of 2-D target grid
    flag_src: 2-D array
        Dimension length is 1 smaller than that of *lat_e_src*
        Only process grid with True flag.
        If flag_src is None, all grids are processed.

    Returns
    -------


    """

    # source dimension
    dim_e_src = lat_e_src.shape
    dim_c_src = (dim_e_src[0] - 1, dim_e_src[1] - 1)
    N_src = dim_c_src[0] * dim_c_src[1]

    # source flag
    if flag_src is None:
        flag_src = np.full(dim_c_src, True)
    else:
        if (flag_src.shape != dim_c_src):
            print(' - calc_overlap_area_record: flag_src dimension error.')
            print('flag_src dimension is: ', flag_src.shape)
            print('dim_c_src dimension is: ', dim_c_src)
            exit()


    # 
    for i_src in range(dim_c_src[0]):
        for j_src in range(dim_c_src[1]):

            # only process data with flag is True
            if  (not flag_src[i_src,j_src]):
                continue

            # find all target grids that overlap with source grid
            lat_corner = [lat_e_src[i_src,j_src], lat_e_src[i_src,j_src+1],
                    lat_e_src[i_src+1,j_src+1], lat_e_src[i_src+1,j_src]]
            lon_corner = [lon_e_src[i_src,j_src], lon_e_src[i_src,j_src+1],
                    lon_e_src[i_src+1,j_src+1], lon_e_src[i_src+1,j_src]]
            limit_src = [min(lat_corner), min(lon_corner),
                    max(lat_corner), max(lon_corner)]
            tar_ind_dict = search_possible_overlap_grid(limit_src, 
                    lat_e_tar, lon_e_tar)
            print(limit_src)
            i0_tar = tar_ind_dict['lat_i0']
            i1_tar = tar_ind_dict['lat_i1']
            j0_tar = tar_ind_dict['lon_j0']
            j1_tar = tar_ind_dict['lon_j1']
            print(lat_e_tar[i0_tar:i1_tar+1,j0_tar:j1_tar+1])
            print(lon_e_tar[i0_tar:i1_tar+1,j0_tar:j1_tar+1])
            print(tar_ind_dict)
            exit()
            
#
#------------------------------------------------------------------------------
#
