animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

settings_dict = {'session': session_settings, 
'ppm': None, # EDIT HERE (will read from pos file)
'smoothing_factor': 3, # EDIT HERE
'useMatchedCut': True, # MUST BE TRUE
'hasObject': True, # EDIT HERE
'plotObject': True, # EDIT HERE
'runFields': True, # EDIT HERE
'plotRate': True, # EDIT HERE
} 


##############################################################################################################################################################################

rate_output = {}
obj_output = {}
centroid_output = {}

keys = ['animal_id','tetrode','unit_id', 'session_ids', 'sliced_wass']
obj_keys = ['animal_id','tetrode','unit_id','session_id', 'obj_pos_x', 'obj_pos_y', 'object_location', 'obj_wass_0', 'obj_wass_90', 'obj_wass_180', 'obj_wass_270', 'obj_wass_no']
centroid_keys = ['animal_id','tetrode','unit_id','session_ids']

for key in keys:
    rate_output[key] = []
for key in obj_keys:
    obj_output[key] = []
for key in centroid_keys:
    centroid_output[key] = []

##############################################################################################################################################################################