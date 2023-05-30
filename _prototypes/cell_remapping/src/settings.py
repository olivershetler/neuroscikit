
""" 
METADATA, IGNORE FOR NOW
"""

animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

settings_dict = {'session': session_settings}

""" 
GLOBAL SETTINGS
"""

settings_dict['session']['channel_count'] = 4
settings_dict['ppm'] = None # EDIT HERE (will auto read from file if None, otherwise will override with this value)
settings_dict['smoothing_factor'] = 2 # EDIT HERE (for plotting)
settings_dict['useMatchedCut'] = False # EDIT HERE (NECESSARY TO BE TRUE OR TO HAVE MANUALLY MATCHED CUT FILES)
settings_dict['n_projections'] = 10**3 # EDIT HERE (10**3 is slow,  50 (default) is faster but less accurate, 10**2 is middle ground --> look paper)
settings_dict['n_shuffle_projections'] = 10**2 # EDIT HERE (10**3 is slow,  50 (default) is faster but less accurate, 10**2 is middle ground --> look paper)
settings_dict['type'] = 'object' # EDIT HERE # Currently only 'object' is supported so no need to change (will add e.g. angle later)
# Type is used to read angle or other (e.g. odor) from filename
##### ratemap size setting (16,16) --> tradeoff between speed and accuracy
settings_dict['ratemap_dims'] = (16,16) # EDIT HERE (16,16) is default, (32,32) is slower but more accurate,
settings_dict['disk_arena'] = False # EDIT HERE. IF TRUE WILL FORCE DISK. IF FALSE WILL CHECK FILE NAME TO SEE IF TRUE OR NOT
settings_dict['normalizeRate'] = True # EDIT HERE --> NORMALIZED FOR ALL CASES 
settings_dict['naming_type'] = 'MEC' # EDIT HERE --> 'MEC' or 'LEC'
settings_dict['rotate_evening'] = False
settings_dict['rotate_angle'] = 90
# sub2 1.xlsx is 32,32 shuffle = 500 with jit - 6062.33 seconds
# sub3 1.xlsx is 32,32 shuffle = 500 no jit - 4550.15 seconds
# sub2 2.xlsx is 32,32 shuffle = 1000 no jit - 13204.84 seconds
# sub3 2.xlss is 16,16 shuffle = 1000 no jit (double check run time with ither sub3 to see which is 1. and 2.) - 7452.62 seconds
# sub4 1.xlsx is 16,16 shuffle = 1000 no jit, vectorized ops - 2315.20 seconds

""" 
IF YOU ARE DOING REGULAR REMAPPING
"""

settings_dict['runRegular'] = True # EDIT HERE
settings_dict['plotRegular'] = True # EDIT HERE
settings_dict['rate_scores'] = ['whole', 'spike_density']
settings_dict['n_repeats'] = 500 # EDIT HERE 
settings_dict['plotShuffled'] = True # EDIT HERE
settings_dict['plotMatchedWaveforms'] = False # EDIT HERE

""" 
IF YOU ARE DOING OBJECT REMAPPING
"""

settings_dict['hasObject'] = False # EDIT HERE
settings_dict['plotObject'] = False # EDIT HERE
settings_dict['object_scores'] = ['whole', 'field', 'binary', 'centroid', 'spike_density']
# settings_dict['grid_sample_threshold'] = 3.2 # EDIT HERE, euclidean distance
settings_dict['spacing'] = 2 # EDIT HERE, same unit as arena height and width
settings_dict['hexagonal'] = True # EDIT HERE, sampling scheme, hexagonal=True or rectangular=True (so hexagonal=False)

settings_dict['downsample'] = False # EDIT HERE
settings_dict['downsample_factor'] = 1 # EDIT HERE

variations = [0,90,180,270,'NO'] # EDIT HERE

""" 
IF YOU ARE DOING CENTROID REMAPPING
"""

settings_dict['runFields'] = False # EDIT HERE
settings_dict['plotFields'] = False # EDIT HERE
settings_dict['centroid_scores'] = ['field', 'binary', 'centroid']

""" 
IF YOU ARE DOING CONTEXT REMAPPING
"""

settings_dict['runUniqueGroups'] = True # EDIT HERE

session_comp_categories = {'morning': [1,3], 'afternoon': [2,4]} # EDIT HERE


##############################################################################################################################################################################
# NO NEED TO EDIT BELOW # 
##############################################################################################################################################################################

regular_output = {}
obj_output = {}
centroid_output = {}

keys = ['signature','depth', 'name', 'date', 'tetrode','unit_id', 'session_ids', 'whole_wass',
        'z_score', 'p_value', 'base_mean', 'base_std', 'mod_z_score', 'mod_p_value', 'median', 'mad', 'spike_density_wass', 'fr_rate', 'fr_rate_ratio', 'fr_rate_change', 
        'n_repeats','arena_size','cylinder','ratemap_dims','downsample_factor']
#  'information', 'b_top', 'b_bottom', 'b_right', 'b_left', 'grid_score']

obj_keys = ['signature','depth', 'name', 'date','tetrode','unit_id','session_id','obj_pos','object_location', 'score', 
            # 'centroid_coords', 'angle', 'magnitude',
            'field_id', 'field_count',
            'obj_wass_0', 'obj_wass_90', 'obj_wass_180', 'obj_wass_270', 'obj_wass_NO', 
            'obj_q_0', 'obj_q_90', 'obj_q_180', 'obj_q_270', 'obj_q_NO',
            'obj_vec_0', 'obj_vec_90', 'obj_vec_180', 'obj_vec_270', 'obj_vec_NO',
            'bin_area', 'total_rate', 'field_peak_rate',
            'field_coverage', 'field_area', 'field_rate', 'cumulative_coverage', 'cumulative_area', 'cumulative_rate',
            'arena_size', 'cylinder', 'ratemap_dims', 'spacing', 'hexagonal', 'sample_size', 'downsample_factor']

centroid_keys = ['signature','depth', 'name', 'date','tetrode','unit_id','session_ids','cumulative_wass',
                 'score', 'field_count', 'bin_area', 'arena_size',
                 'cumulative_coverage', 'cumulative_area', 'cumulative_rate',
                 'arena_size', 'cylinder', 'ratemap_dims']
# 'test_wass','centroid_wass','binary_wass']

for key in keys:
    regular_output[key] = []
for key in obj_keys:
    obj_output[key] = []
for key in centroid_keys:
    centroid_output[key] = []

##############################################################################################################################################################################

task_keys = ['binary_map', 'autocorrelation_map', 'sparsity', 'selectivity', 'information', 'coherence', 'speed_score', 'hd_score', 'tuning_curve', 'grid_score', 'border_score', 'field_sizes', 'disk_arena']
# true_tasks = ['information', 'border_score', 'grid_score']
true_tasks = []
tasks = {}
for key in task_keys:
    if key == 'disk_arena':
        tasks[key] = False
    elif key in true_tasks:
        tasks[key] = True
    else:   
        tasks[key] = False

##############################################################################################################################################################################

# session_comp_categories = None
context_output = {}
afternoon_output = {}
morning_output = {}

if settings_dict['runUniqueGroups'] == True:

    # session_comp_categories = {'morning': [1,3], 'afternoon': [2,4]}
    # keys = ['signature','depth','name','date','tetrode','unit_id', 'session_ids', 'sliced_wass']
    keys = ['signature','depth', 'name', 'date', 'tetrode','unit_id', 'session_ids', 'whole_wass',
            'z_score', 'p_value', 'base_mean', 'base_std', 'mod_z_score', 'mod_p_value', 'median', 'mad', 'spike_density_wass', 'fr_rate', 'fr_rate_ratio', 'fr_rate_change', 
            'n_repeats','arena_size','cylinder','ratemap_dims','downsample_factor']

    for key in keys:
        morning_output[key] = []
    context_output['morning'] = morning_output

    for key in keys:
        afternoon_output[key] = []
    context_output['afternoon'] = afternoon_output