import numpy as np

def fit_time_regs_to_cells(sequential_trials, time_regressors, settings_dict):
    # full_cell_time_regs = np.zeros((3, int(len(settings_dict['trial_start_times']) * settings_dict['trial_length']/settings_dict['bin_timestep'])))

    time_regs_fit_to_cells = list(map(lambda j: _fit_time_regs_to_single_cell(j, sequential_trials, time_regressors, settings_dict), np.arange(len(sequential_trials))))

    return time_regs_fit_to_cells

def _fit_time_regs_to_single_cell(j, sequential_trials, time_regressors, settings_dict):

    full_exp_length = int(len(sequential_trials[j]) * len(settings_dict['trial_start_times']) * settings_dict['trial_length']/settings_dict['bin_timestep'])
    full_cell_time_regs = np.zeros((3, full_exp_length))

    ramp_regs = list(map(lambda i: _concat_time_regressors(i, sequential_trials[j], time_regressors, full_cell_time_regs, settings_dict['trial_start_times']), np.arange(len(time_regressors))))

    full_cell_time_regs[0,:] = ramp_regs[0]
    full_cell_time_regs[1,:] = ramp_regs[1]

    # make full experiment ramp
    int(len(settings_dict['trial_start_times']) * settings_dict['trial_length']/settings_dict['bin_timestep'])
    dt = settings_dict['bin_timestep']
    experiment_ramp = np.arange(0, full_exp_length, 1) * dt
    # experiment_ramp = experiment_ramp/np.linalg.norm(experiment_ramp)
    # experiment_ramp = experiment_ramp[:int(len(sequential_trials[j]))]

    # full_cell_time_regs.append(experiment_ramp)
    full_cell_time_regs[-1,:] = experiment_ramp

    return full_cell_time_regs

def _concat_time_regressors(i, cell_trials_sequences, time_regressors, full_cell_time_regs, trial_start_times):
    ses_ct = len(cell_trials_sequences)

    for k in range(ses_ct):
        if k == 0:
            ramp_reg = time_regressors[i]
        else:
            ramp_reg = np.hstack((ramp_reg, time_regressors[i]))

    return ramp_reg

def collect_cell_endog_exog(sequential_trials, regressors: list | tuple | np.ndarray, settings_dict):
    X = []
    y = []
        

    for i in range(len(sequential_trials)):
        cell_y = np.concatenate(sequential_trials[i])
        cell_X = []
        print('new')
        for reg_key in regressors:
            reg = regressors[reg_key]
            
            if reg_key == 'odor':
                cell_pulses = reg[i]
                if settings_dict['pulse_method'] == 'single':
                    if j == 0:
                        cell_X = np.asarray(cell_pulses.reshape((1,-1)))
                    else:
                        cell_X = np.vstack((cell_X, np.asarray(cell_pulses[j]).reshape((1,-1))))
                elif settings_dict['pulse_method'] == 'multi':
                    for j in range(len(cell_pulses)):
                        if j == 0 and len(cell_X) == 0:
                            cell_X = np.asarray(cell_pulses[j]).reshape((1,-1))
                        else:
                            cell_X = np.vstack((cell_X, np.asarray(cell_pulses[j]).reshape((1,-1))))

            if reg_key == 'time':
                time_regressors = reg[i]
                for j in range(len(time_regressors)):
                    if j == 0 and len(cell_X) == 0:
                        cell_X = np.asarray(time_regressors[j]).reshape((1,-1))
                    else:
                        cell_X = np.vstack((cell_X, np.asarray(time_regressors[j]).reshape((1,-1))))

        X.append(cell_X)
        y.append(cell_y)

    return y, X
