
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
settings_dict['ppm'] = None # EDIT HERE
settings_dict['smoothing_factor'] = 2 # EDIT HERE
settings_dict['useMatchedCut'] = False # EDIT HERE

""" 
IF YOU ARE DOING RATE REMAPPING
"""

settings_dict['plotRate'] = False # EDIT HERE
settings_dict['normalizeRate'] = True # EDIT HERE

""" 
IF YOU ARE DOING OBJECT REMAPPING
"""

settings_dict['hasObject'] = False # EDIT HERE
settings_dict['plotObject'] = False # EDIT HERE

variations = [0,90,180,270,'no'] # EDIT HERE

""" 
IF YOU ARE DOING CENTROID REMAPPING
"""

settings_dict['runFields'] = True # EDIT HERE
settings_dict['plotFields'] = False # EDIT HERE

""" 
IF YOU ARE DOING CONTEXT REMAPPING
"""

settings_dict['runUniqueGroups'] = True # EDIT HERE

session_comp_categories = {'morning': [1,3], 'afternoon': [2,4]} # EDIT HERE


##############################################################################################################################################################################
# NO NEED TO EDIT BELOW # 
##############################################################################################################################################################################

rate_output = {}
obj_output = {}
centroid_output = {}

keys = ['animal_id','tetrode','unit_id', 'session_ids', 'sliced_wass', 'bin_wass']
#  'information', 'b_top', 'b_bottom', 'b_right', 'b_left', 'grid_score']
obj_keys = ['animal_id','tetrode','unit_id','session_id', 'obj_pos_x', 'obj_pos_y', 'object_location', 'obj_wass_0', 'obj_wass_90', 'obj_wass_180', 'obj_wass_270', 'obj_wass_no']
centroid_keys = ['animal_id','tetrode','unit_id','session_ids','cumulative_wass']
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
    keys = ['animal_id','tetrode','unit_id', 'session_ids', 'sliced_wass']

    for key in keys:
        morning_output[key] = []
    context_output['morning'] = morning_output

    for key in keys:
        afternoon_output[key] = []
    context_output['afternoon'] = afternoon_output