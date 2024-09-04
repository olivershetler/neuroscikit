# Outside imports
import os, sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import time

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.LEC_analysis.src.settings import settings_dict
from x_io.rw.axona.batch_read import make_study
from _prototypes.LEC_analysis.src.format import aggregate_event_times_matched, sequence_cell_trials
from _prototypes.LEC_analysis.src.regressors import make_cell_regressors
from _prototypes.LEC_analysis.src.glm import batchStudyGLM
import matplotlib.pyplot as plt


def main():
    start_time = time.time()
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root,title='Please select a data directory.')
    output_path = data_dir + '/output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    study = make_study(data_dir,settings_dict)
    study.make_animals()

    agg_events, agg_event_objects, agg_events_binary, agg_trial_types = aggregate_event_times_matched(study, settings_dict)

    sequential_trials, sequential_labels, trial_dict, new_time_index = sequence_cell_trials(agg_events, agg_event_objects, agg_trial_types, settings_dict)

    pulses, time_regs, _ = make_cell_regressors(sequential_trials, sequential_labels, new_time_index, settings_dict)

    print(np.array(pulses).shape, np.array(pulses)[0].shape, len(np.array(pulses)[0,0]), len(np.array(pulses)[1,0]), len(np.array(pulses)[0,1]), len(np.array(pulses)[0,2]), np.array(time_regs).shape)

    y, X = collect_cell_endog_exog(sequential_trials, pulses, time_regs, settings_dict)

    # print(synth_regressors.shape, sequential_trials.shape, len(synth_regressors[0]), len(sequential_trials[0]), len(synth_regressors[0][0]), len(sequential_trials[0][0]))
    # study_inp_data = formatDataGLM()

    # normX = X / np.linalg.norm(X)
    # print('starting', y.shape, X.shape)

    glm_results = batchStudyGLM(y, X, settings_dict['GLM_kwargs'])

    for res_arr in glm_results:
        print(res_arr[0])

        # fig = plt.figure(figsize=(12,5))

        # ax = plt.subplot(1,1,1)
        # ax.plot(res_arr[1])

        # ax2 = ax.twinx()

        # ax2.plot(rese_arr)

def collect_cell_endog_exog(sequential_trials, pulses, time_regs, settings_dict):
    X = []
    y = []
        

    for i in range(len(sequential_trials)):
        cell_y = sequential_trials[i]
        cell_pulses = pulses[i]

        # time regreessors are seession and trial ramp, session ramp is ramp across all trials in session (e.g. with 4 odor trials its one ramp every 4 trials for session)
        full_cell_time_regs = np.zeros((3, len(cell_pulses[0])))
        # print(time_regs)
        for i in range(len(time_regs)):
            ramp_reg = []
            if settings_dict['pulse_method'] == 'single':
                ct = int(len(cell_pulses)/len(time_regs[i]))
            elif settings_dict['pulse_method'] == 'multi': 
                ct = int(len(cell_pulses[0])/len(time_regs[i]))
            # print(len(cell_pulses)/len(time_regs[i]), len(cell_pulses), len(time_regs[i]))
            for j in range(ct):
                if j == 0:
                    ramp_reg = time_regs[i]
                else:
                    ramp_reg = np.hstack((ramp_reg, time_regs[i]))

            # ramp_reg = ramp_reg/np.linalg.norm(ramp_reg)
            print(ramp_reg.shape)
            # full_cell_time_regs.append(ramp_reg)
            full_cell_time_regs[i,:] = ramp_reg
        
        # make full experiment ramp
        # dt = time_regs[0][1] - time_regs[0][0]
        dt = settings_dict['bin_timestep']
        experiment_ramp = np.arange(0, float(len(time_regs[0]) * dt * ct), dt)
        # experiment_ramp = experiment_ramp/np.linalg.norm(experiment_ramp)
        print('READ HERE')
        print(ct, len(cell_pulses[0]), len(time_regs[i]))
        print(experiment_ramp.shape,experiment_ramp[-1],full_cell_time_regs[0][-1],full_cell_time_regs[1][-1])
        experiment_ramp = experiment_ramp[:len(ramp_reg)]
        # full_cell_time_regs.append(experiment_ramp)
        full_cell_time_regs[-1,:] = experiment_ramp

        full_cell_time_regs = np.array(full_cell_time_regs)
        print(full_cell_time_regs.shape)
        print(full_cell_time_regs)


        cell_X = []
        if settings_dict['pulse_method'] == 'single':
            cell_X = np.vstack((full_cell_time_regs, cell_pulses.reshape((1,-1))))
        elif settings_dict['pulse_method'] == 'multi':
            for j in range(len(cell_pulses)):
                # print(np.array(cell_pulses[j].shape))
                if j == 0:
                    cell_X = np.asarray(cell_pulses[j]).reshape((1,-1))
                else:
                    cell_X = np.vstack((cell_X, np.asarray(cell_pulses[j]).reshape((1,-1))))

            cell_X = np.vstack((full_cell_time_regs, cell_X))

        X.append(cell_X)
        y.append(cell_y)

        print('UNIQUE')
        print(np.unique(cell_X))
        print(cell_X.shape)

    return y, X




if __name__ == '__main__':
    main()