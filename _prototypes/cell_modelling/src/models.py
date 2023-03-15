import os, sys
import numpy as np


PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.cell_modelling.src.format_neural import aggregate_spike_objects, aggregate_spike_counts, aggregate_spike_times, aggregate_trial_types, sequence_cell_trials
from _prototypes.cell_modelling.src.regressors import make_time_regressors, make_label_regressors
from _prototypes.cell_modelling.src.format_glm import fit_time_regs_to_cells, collect_cell_endog_exog
from _prototypes.cell_modelling.src.glm import batchStudyGLM

def get_models(study, settings_dict, subdir):
    for animal in study.animals:
        print(settings_dict.keys())

        if 'GLM' in settings_dict['models_to_run']:

            spike_times = aggregate_spike_times(animal)
            trial_types = aggregate_trial_types(animal, settings_dict)
            regressors = {}

            sequential_trials, sequential_labels, new_time_index = sequence_cell_trials(spike_times, trial_types, settings_dict)
            time_index_bools = list(map(lambda x: list(map(lambda y: np.array_equiv(new_time_index[x], new_time_index[y]), np.arange(len(new_time_index)))), np.arange(len(new_time_index))))
            assert False not in np.unique(time_index_bools)
            new_time_index = new_time_index[0]

            if 'time' in settings_dict['regressors_to_use']['GLM']:
                # session ramp, trial ramp (experiment ramp added in next step)
                time_regressors = make_time_regressors(new_time_index, settings_dict['trial_start_times'], settings_dict['trial_length'], settings_dict['addInterTrial'])
                X_time = fit_time_regs_to_cells(sequential_trials, time_regressors, settings_dict)
                regressors['time'] = X_time

            if 'odor' in settings_dict['regressors_to_use']['GLM']:
                X_odor = []
                for i in range(len(sequential_labels)):
                    cell_pulses = make_label_regressors(new_time_index, sequential_labels[i], settings_dict['p_start'], settings_dict['p_window'], settings_dict['trial_length'], method=settings_dict['pulse_method'])
                    X_odor.append(cell_pulses)
                regressors['odor'] = X_odor

            y, X = collect_cell_endog_exog(sequential_trials, regressors, settings_dict)

            glm_results = batchStudyGLM(y, X, settings_dict['GLM_kwargs'])


        # make_spike_lfp_regressors()
        
