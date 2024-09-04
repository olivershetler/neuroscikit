""" If a setting is not used for your analysis (e.g. smoothing_factor), just pass in an arbitrary value or pass in 'None' """
STUDY_SETTINGS = {

    'ppm': 511,  # EDIT HERE

    'smoothing_factor': None, # EDIT HERE

    'useMatchedCut': True,  # EDIT HERE
}


# Switch devices to True/False based on what is used in the acquisition (to be extended for more devices in future)
device_settings = {'axona_led_tracker': True, 'implant': True} 

# Make sure implant metadata is correct, change if not, AT THE MINIMUM leave implant_type: tetrode
implant_settings = {'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

# WE ASSUME DEVICE AND IMPLANT SETTINGS ARE CONSISTENCE ACROSS SESSIONS

# Set channel count + add device/implant settings
SESSION_SETTINGS = {
    'channel_count': 4, # EDIT HERE, default is 4, you can change to other number but code will check how many tetrode files are present and set that to channel copunt regardless
    'devices': device_settings, # EDIT HERE
    'implant': implant_settings, # EDIT HERE
}

STUDY_SETTINGS['session'] = SESSION_SETTINGS

settings_dict = STUDY_SETTINGS


# Pre-process args - ALL IN SECONDS
settings_dict['bin_timestep'] = 3
settings_dict['trial_start_times'] = [0,60,120,180]
settings_dict['trial_length'] = 60
 
settings_dict['p_start'] = 0 # odor/object presentation onset
settings_dict['p_window'] = 10 # duration of presentation

# Custom
settings_dict['firstIsBaseline'] = True
# assumes first is baseline
settings_dict['addInterTrial'] = False
settings_dict['pulse_method'] = 'multi' # 'single' is one reg for all odors/obj, 'multi' is 1/0 (present/not present) encoding with as many pulses as odors/obj

# GLM kwargs
settings_dict['GLM_kwargs'] = {
    'p': 0.8,
    'model': 'statsmodels', 
    'link': 'Poisson', 
    'custom_intercept': False
}