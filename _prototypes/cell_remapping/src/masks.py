import os, sys
import numpy as np
import re

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.maps.map_utils import _interpolate_matrix, disk_mask


def make_object_ratemap(object_location, rate_map_obj, new_size=(16,16)):
    #arena_height, arena_width = rate_map_obj.arena_size
    #arena_height = arena_height[0]
    #arena_width = arena_width[0]

    rate_map, _ = rate_map_obj.get_rate_map(new_size=new_size)

    # (64, 64)
    y, x = rate_map.shape
    # print(y, x)
    # convert height/width to arrayswith 64 bins
    #height = np.arange(0,arena_height, arena_height/x)
    #width = np.arange(0,arena_width, arena_width/y)

    # make zero array same shape as true ratemap == fake ratemap
    arena = np.zeros((y,x))

    # if no object, zero across all ratemap
    if object_location == 'NO':
        cust_arena = np.ones((y,x))
        norm_arena = cust_arena / np.sum(cust_arena)
        return norm_arena, {'x':0, 'y':0}

    else:
        # if object, pass into dictionary to get x/y coordinates of object location
        # object_location_dict = {
        #     0: (y-1, int(np.floor(x/2))),
        #     90: (int(np.floor(y/2)), x-1),
        #     180: (0, int(np.floor(x/2))),
        #     270: (int(np.floor(y/2)), 0)
        # }

        object_location_dict = {
            0: (0,int(np.floor(x/2))),
            90: (int(np.floor(y/2)),y-1),
            180: (y-1,int(np.floor(x/2))),
            270:  (int(np.floor(y/2)),0)
        }

        id_y, id_x = object_location_dict[object_location]

        # get x and y ids for the first bin that the object location coordinates fall into
        #id_x = np.where(height <= object_pos[0])[0][-1]
        #id_y = np.where(width <= object_pos[1])[0][-1]
        # id_x_small = np.where(height < object_pos[0])[0][0]



        # cts_x, _ = np.histogram(object_pos[0], bins=height)
        # cts_y, _ = np.histogram(object_pos[1], bins=width)

        # id_x = np.where(cts_x != 0)[0]
        # id_y = np.where(cts_y != 0)[0]
        # print(arena_height, arena_width, height, width, object_pos, id_x, id_y)

        # set that bin equal to 1

        arena[id_y, id_x] = 1

        # print(np.max(rate_map), np.max(rate_map)-np.min(rate_map), np.min(rate_map), np.sum(rate_map))
        # arena[id_x, id_y] = np.sum(rate_map)

        # print(arena)

        return arena, {'x':id_x, 'y':id_y}


def check_disk_arena(path):
    variations = [r'cylinder', r'round', r'circle', r'CYLINDER', r'ROUND', r'CIRCLE', r'Cylinder', r'Round', r'Circle']
    var_bool = []
    true_var = None
    for var in variations:
        if re.search(var, path) is not None:
            var_bool.append(True)
            true_var = var
        else:
            var_bool.append(False)
    # if re.search(r'cylinder', path) is not None or re.search(r'round', path) is not None:
    if np.array(var_bool).any() == True:
        cylinder = True
    else:
        cylinder = False
    # print(cylinder, true_var, path)
    return cylinder, true_var


def flat_disk_mask(rate_map):
    masked_rate_map = disk_mask(rate_map)
    # masked_rate_map.data[masked_rate_map.mask] = 0
    # masked_rate_map.data[masked_rate_map.mask] = np.nan
    # print(np.unique(masked_rate_map.data))
    # return  masked_rate_map.data
    copy = np.copy(rate_map).astype(np.float32)
    copy[masked_rate_map.mask] = np.nan
    return copy
    # return masked_rate_map
