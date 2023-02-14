""" If a setting is not used for your analysis (e.g. smoothing_factor), just pass in an arbitrary value or pass in 'None' """
STUDY_SETTINGS = {

    'ppm': 511,  # EDIT HERE

    'smoothing_factor': None, # EDIT HERE

    'useMatchedCut': True,  # EDIT HERE, set to False if you want to use runUnitMatcher, set to True after to load in matched.cut file
}


# Switch devices to True/False based on what is being used (to be extended for more devices in future)
device_settings = {'axona_led_tracker': False, 'implant': True}
# make_session will fail if axona_led_tracker is True and no led_tracker file (.pos) is found in the session folder for any of the groups of session files.

# Make sure implant metadata is correct, change if not, AT THE MINIMUM leave implant_type: tetrode
implant_settings = {'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

# WE ASSUME DEVICE AND IMPLANT SETTINGS ARE CONSISTENCE ACROSS SESSIONS, IF THIS IS NOT THE CASE PLEASE LET ME KNOW

# Set channel count + add device/implant settings
SESSION_SETTINGS = {
    'channel_count': 4, # EDIT HERE, default is 4, you can change to other number but code will check how many tetrode files are present and set that to channel copunt regardless
    'devices': device_settings, # EDIT HERE
    'implant': implant_settings, # EDIT HERE
}

STUDY_SETTINGS['session'] = SESSION_SETTINGS

settings_dict = STUDY_SETTINGS

