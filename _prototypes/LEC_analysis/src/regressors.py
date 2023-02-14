import numpy as np
import os,sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import matplotlib.pyplot as plt


def make_cell_regressors(sequential_trials, sequential_labels, time_index, settings_dict):
    
    session_ramping, trial_ramping = get_time_regressors(time_index, settings_dict['trial_start_times'], settings_dict['trial_length'], settings_dict['addInterTrial'])

    pulses = []
    for i in range(len(sequential_labels)):
        cell_pulses = get_label_regressors(time_index, sequential_labels[i], settings_dict['p_start'], settings_dict['p_window'], settings_dict['trial_length'], method=settings_dict['pulse_method'])
        print(np.array(cell_pulses).shape, len(cell_pulses[0]))
        pulses.append(cell_pulses)


    print(np.array(session_ramping).shape, np.array(trial_ramping).shape)
    time_regs = np.vstack((session_ramping.reshape((1,-1)), trial_ramping.reshape((1,-1))))
    # print(time_regs.shape, np.array(pulses).shape)
    # synth_regressors = np.vstack((pulses, synth_regressors))

    bio_regressors = None
    # to do ...

    return pulses, time_regs, bio_regressors

def get_time_regressors(time_index, trial_start_times, trial_length, addInterTrial):

    dt = time_index[1] - time_index[0]

    if addInterTrial:
        session_ramping = np.arange(0, len(time_index)* dt * (len(trial_start_times)+2), dt)
    else:
        session_ramping = np.arange(0, len(time_index)* dt * len(trial_start_times), dt)

    print(len(time_index)/dt, len(trial_start_times), dt)
    trial_ramping = []
    for i in range(len(trial_start_times)):
        ramp = time_index
        trial_ramping = np.hstack((trial_ramping, ramp))
        if i == 0:
            prev_ramp = ramp
        if addInterTrial and i != 0 and i != len(trial_start_times)-1:
            trial_ramping = np.hstack((trial_ramping, prev_ramp))

    # if len(trial_ramping) < len(session_ramping):
    #     trial_ramping = np.hstack((trial_ramping, np.arange(0, session_ramping[-1]-trial_start_times[-1]-trial_length+dt, dt)))

    # # NORMALIZE
    # trial_ramping = trial_ramping / np.sum(trial_ramping)
    # session_ramping = session_ramping / np.sum(session_ramping)

    return session_ramping, trial_ramping

def get_label_regressors(time_index, labels, p_start, p_window, trial_length, method='multi'):

    unq_labels = np.unique(labels)

    if method == 'multi':
        pulses = []
        for i in range(len(unq_labels)):
            pulses.append(np.zeros(int(len(time_index)*len(labels))))
    elif method =='single':
        pulses = np.zeros(len(time_index))

    start = np.where(np.array(time_index) >= float(p_start))[0][0]
    end = np.where(np.array(time_index) <= float(p_start + p_window))[0][-1]
    # time index is bins of binned count (fron np.histogram), specifically its the starting bin 
    trial_end = np.where(np.array(time_index) <= float(p_start + trial_length))[0][-1] + 1

    for i in range(len(labels)):
        lbl_id = np.where(labels[i] == unq_labels)[0][0]
        print('aqui')
        print(lbl_id, labels[i], unq_labels)

        # print(labels[i], unq_labels, lbl_id, np.where((labels[i] == unq_labels)))

        trial_pulse = np.zeros(time_index.shape)
        
        if labels[i] != 'o':
            trial_pulse[start:end] = 1

        print(p_start, p_window, trial_length, time_index.shape)
        print(start, end ,trial_end, start + int(i * time_index.shape[0]), trial_end + int(i * time_index.shape[0]))
        if method == 'multi':
            pulses[lbl_id][start + int(i * time_index.shape[0]):trial_end + int(i * time_index.shape[0])] = trial_pulse
            
        elif method =='single':
            trial_pulse *= lbl_id
            pulses = np.hstack((pulses, trial_pulse))

    # # NORMALIZE
    # if method == 'multi':
    #     for unq in range(len(pulses)):
    #         pulses[unq]  = pulses[unq] / np.linalg.norm(pulses[unq])
    # elif method == 'single':
    #     pulses = pulses / np.linalg.norm(pulses)

    # for reg in pulses:
    #     fig = plt.figure(figsize=(8,3))
    #     plt.plot(reg)
    #     plt.show()


    return pulses