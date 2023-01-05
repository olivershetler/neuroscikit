import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.waveform_plotter.src.plots import plot_cell_waveform



def batch_plots(study, settings_dict, data_dir):

    output_path = data_dir + '/output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for animal in study.animals:

        output_path = data_dir + '/output/' + str(animal.animal_id) + '/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for session_key in animal.sessions:
            session = animal.sessions[session_key]

            output_path = data_dir + '/output/' + str(animal.animal_id) + '/' + str(session_key) + '/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            for cell in session.get_cell_data()['cell_ensemble'].cells:

                if settings_dict['plotCellWaveforms']:

                    plot_cell_waveform(cell, data_dir)