
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
settings_dict['smoothing_factor'] = 3 # EDIT HERE (for plotting)
settings_dict['useMatchedCut'] = True # EDIT HERE (NECESSARY TO BE TRUE OR TO HAVE MANUALLY MATCHED CUT FILES)
settings_dict['n_projections'] = 10**2 # EDIT HERE (10**3 is slow,  50 (default) is faster but less accurate, 10**2 is middle ground --> look paper)
settings_dict['type'] = 'object' # EDIT HERE # Currently only 'object' is supported so no need to change (will add e.g. angle later)
# Type is used to read angle or other (e.g. odor) from filename
##### ratemap size setting (16,16) --> tradeoff between speed and accuracy
settings_dict['ratemap_dims'] = (64,64) # EDIT HERE (16,16) is default, (32,32) is slower but more accurate,

""" 
IF YOU ARE DOING WHOLE MAP REMAPPING
"""

settings_dict['plotRate'] = True # EDIT HERE
settings_dict['normalizeRate'] = True # EDIT HERE --> NORMALIZED FOR ALL CASES 
settings_dict['rate_scores'] = ['whole', 'binary']


""" 
IF YOU ARE DOING OBJECT REMAPPING
"""

settings_dict['hasObject'] = True # EDIT HERE
settings_dict['plotObject'] = True # EDIT HERE
settings_dict['object_scores'] = ['whole', 'field', 'binary', 'centroid']

variations = [0,90,180,270,'NO'] # EDIT HERE

""" 
IF YOU ARE DOING CENTROID REMAPPING
"""

settings_dict['runFields'] = True # EDIT HERE
settings_dict['plotFields'] = True # EDIT HERE
settings_dict['centroid_scores'] = ['field', 'binary', 'centroid']

""" 
IF YOU ARE DOING CONTEXT REMAPPING
"""

settings_dict['runUniqueGroups'] = False # EDIT HERE

session_comp_categories = {'morning': [1,3], 'afternoon': [2,4]} # EDIT HERE


##############################################################################################################################################################################
# NO NEED TO EDIT BELOW # 
##############################################################################################################################################################################

rate_output = {}
obj_output = {}
centroid_output = {}

keys = ['signature','depth', 'name', 'date', 'tetrode','unit_id', 'session_ids', 'score', 'wass']
#  'information', 'b_top', 'b_bottom', 'b_right', 'b_left', 'grid_score']

obj_keys = ['signature','depth', 'name', 'date','tetrode','unit_id','session_id','obj_pos_x', 'obj_pos_y', 'object_location', 'score', 
            'obj_wass_0', 'obj_wass_90', 'obj_wass_180', 'obj_wass_270', 'obj_wass_NO', 'field_count', 'bin_area',
            'main_field_coverage', 'main_field_area', 'main_field_rate', 'cumulative_coverage', 'cumulative_area', 'cumulative_rate',]

centroid_keys = ['signature','depth', 'name', 'date','tetrode','unit_id','session_ids','cumulative_wass',
                 'score', 'field_count', 'bin_area', 
                 'cumulative_coverage', 'cumulative_area', 'cumulative_rate']
# 'test_wass','centroid_wass','binary_wass']

for key in keys:
    rate_output[key] = []
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
    keys = ['signature','depth','name','date','tetrode','unit_id', 'session_ids', 'sliced_wass']

    for key in keys:
        morning_output[key] = []
    context_output['morning'] = morning_output

    for key in keys:
        afternoon_output[key] = []
    context_output['afternoon'] = afternoon_output