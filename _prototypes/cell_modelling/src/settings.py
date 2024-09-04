



animal = {'animal_id': '001', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': '001', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

""" FOR YOU TO EDIT """
settings_dict = {'ppm': 511, 'session':  session_settings, 'smoothing_factor': 3, 'useMatchedCut': True}
""" FOR YOU TO EDIT """

# Pre-process args - ALL IN SECONDS
settings_dict['bin_timestep'] = 3 # Time window to use for binning data and making synth regressors
settings_dict['trial_start_times'] = [0,60,120,180] # trial start times
settings_dict['trial_length'] = 60 # length of trial within sessions
settings_dict['p_start'] = 0 # stimulus presentation onset
settings_dict['p_window'] = 10 # duration of stimulus presentation

# Custom
settings_dict['baseline_trial_ids'] = [0] # trial count from 0 to N corresponding to baseline trials (if consistent across sessions)
settings_dict['addInterTrial'] = False # interweave baseline trials in between stimulus trials
settings_dict['pulse_method'] = 'multi' 
settings_dict['firstIsBaseline'] = True
# 'single' is one regressors for all odors/obj
# 'multi' is 1/0 (present/not present) encoding with as many pulse regressors as odors/obj

# LIST OF MODELS YOU WANT TO BE RUN - make sure to edit kwargs for each possible model
settings_dict['models_to_run'] = ['GLM',]
# ONE LIST PER MODEEL
settings_dict['regressors_to_use'] = {'GLM' : ['time','odor'],}

# Generalized linear model
# Possible models: statsmodels, sklearn, 
settings_dict['GLM_kwargs'] = {
    'split': 0.9, # TRAIN TEST SPLIT
    'model': 'statsmodels', 
    'link': 'Poisson', 
    'custom_intercept': False,
    'regularized': False, # 
    'l_split': 0.5, # l1 (lasso) /l2 (ridge) split, only used if regularized = True
}


# Generalized linear mixed model

# Linear regression
# Possible models: statsmodels, sklearn, 

# Non-linear regressions

# Non-linear machine learning (NN)



# CFC

# phast-amplitude

# spike LFP