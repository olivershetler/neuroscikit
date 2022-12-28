animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

settings_dict = {'ppm': 511, 'session': session_settings, 'smoothing_factor': 3, 'useMatchedCut': True, 'hasObject': True}


##############################################################################################################################################################################

rate_output = {}
obj_output = {}
keys = ['animal_id','tetrode','unit_id','wass', 'session_ids', 'sliced_wass']
obj_keys = ['animal_id','tetrode','unit_id','session_id', 'obj_pos_x', 'obj_pos_y', 'object_location', 'obj_wass_0', 'obj_wass_90', 'obj_wass_180', 'obj_wass_270', 'obj_wass_no']
for key in keys:
    rate_output[key] = []
for key in obj_keys:
    obj_output[key] = []

##############################################################################################################################################################################