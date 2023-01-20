import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.waveform_plotter.src.plots import plot_cell_waveform, plot_cell_rate_map



def batch_plots(study, settings_dict, data_dir):

    output_path = data_dir + '/output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for animal in study.animals:

        if settings_dict['outputStructure'] == 'nested' or settings_dict['outputStructure'] == 'sequential':
            output_path = data_dir + '/output/' + str(animal.animal_id) + '/'
        elif settings_dict['outputStructure'] =='single':
            split = str(animal.animal_id).split('tet')
            animal_id = split[0]
            tet_id = split[-1]
            output_path = data_dir + '/output/' 
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        

        for session_key in animal.sessions:
            session = animal.sessions[session_key]

            if settings_dict['outputStructure'] == 'nested' or settings_dict['outputStructure'] == 'sequential':
                output_path = data_dir + '/output/' + str(animal.animal_id) + '/' + str(session_key) + '/'
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

            for cell in session.get_cell_data()['cell_ensemble'].cells:

                if settings_dict['plotCellWaveforms']:
                    
                    if settings_dict['outputStructure'] == 'nested' or settings_dict['outputStructure'] == 'sequential':
                        output_path = data_dir + '/output/' + str(animal.animal_id) + '/' + str(session_key) + '/waveforms/'
                    elif settings_dict['outputStructure'] == 'single':
                        output_path = data_dir + '/output/waveforms/'
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    plot_cell_waveform(cell, output_path)

                if settings_dict['plotCellRatemap']:

                    if settings_dict['outputStructure'] == 'nested' or settings_dict['outputStructure'] == 'sequential':
                        output_path = data_dir + '/output/' + str(animal.animal_id) + '/' + str(session_key) + '/ratemaps/'
                    elif settings_dict['outputStructure'] == 'single':
                        output_path = data_dir + '/output/ratemaps/'
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    plot_cell_rate_map(cell, output_path)

