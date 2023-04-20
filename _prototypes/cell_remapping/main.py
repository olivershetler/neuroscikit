import os, sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import time
import traceback

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


from _prototypes.cell_remapping.src.settings import settings_dict
from x_io.rw.axona.batch_read import make_study
from _prototypes.cell_remapping.src.remapping import compute_remapping


def main(overwrite_settings=None):
    start_time = time.time()
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root,title='Please select a data directory.')

    """ OPTION 1 """
    """ RUNS EVERYTHING UNDER PARENT FOLDER (all subfolders loaded first) """
    # if not os.path.isdir(subdir + '/output'):
    #     os.mkdir(data_dir + '/output')
    # study = make_study(data_dir,settings_dict=settings_dict)
    # study.make_animals()
    # if overwrite_settings is not None:
    #     output = compute_remapping(study, overwrite_settings, data_dir)
    # else:
    #     output = compute_remapping(study, settings_dict, data_dir)
    # _save_output(output, data_dir, start_time)


    """ OPTION 2 """
    """ RUNS EACH SUBFOLDER ONE AT A TIME """
    subdirs = np.sort([ f.path for f in os.scandir(data_dir) if f.is_dir() ])
    count = 1
    for subdir in subdirs:
        try:
            # if subdir/output exists
            if not os.path.isdir(subdir + '/output'):
                os.mkdir(subdir + '/output')
                print('here')
            study = make_study(subdir,settings_dict=settings_dict)
            study.make_animals()
            if overwrite_settings is not None:
                output = compute_remapping(study, overwrite_settings, subdir)
            else:
                output = compute_remapping(study, settings_dict, subdir)
            _save_output(output, subdir, start_time)
            count += 1
        except Exception:
            print(traceback.format_exc())
            print('DID NOT WORK FOR DIRECTORY ' + str(subdir))


def _save_output(output, output_path, start_time):

    # for ky in output['centroid']:
    #     print(ky)
    #     print(len(output['centroid'][ky]))

    if 'regular' in output:
        df = pd.DataFrame(output['regular'])
        # df.to_csv(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/rate_remapping.csv')
        df.to_excel(output_path + '/output/regular_remapping.xlsx')
    if 'object' in output:
        df = pd.DataFrame(output['object'])
        # df.to_csv(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/obj_remapping.csv')
        df.to_excel(output_path + '/output/obj_remapping.xlsx')
    if 'centroid' in output:
        df = pd.DataFrame(output['centroid'])
        # df.to_csv(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/centroid_remapping.csv')
        df.to_excel(output_path + '/output/centroid_remapping.xlsx')
    if 'context' in output:
        for context in output['context']:
            df = pd.DataFrame(output['context'][context])
            # df.to_csv(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/context_output.csv')
            df.to_excel(output_path + '/output/' + str(context) + '_output.xlsx')

    print('Total run time: ' + str(time.time() - start_time))

if __name__ == '__main__':
    main()

