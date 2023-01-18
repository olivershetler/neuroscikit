import os, sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import time

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


from _prototypes.cell_remapping.src.settings import settings_dict
from x_io.rw.axona.batch_read import make_study
from _prototypes.cell_remapping.src.remapping import compute_remapping


def main():
    start_time = time.time()
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root,title='Please select a data directory.')
    study = make_study(data_dir,settings_dict)
    study.make_animals()
    output = compute_remapping(study, settings_dict)
    for key in output['centroid']:
        print(key)
        print(np.array(output['centroid'][key]).shape)
        print(output['centroid'][key])
    if 'rate' in output:
        df = pd.DataFrame(output['rate'])
        # df.to_csv(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/rate_remapping.csv')
        df.to_excel(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/rate_remapping.xlsx')
    if 'object' in output:
        df = pd.DataFrame(output['object'])
        # df.to_csv(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/obj_remapping.csv')
        df.to_excel(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/obj_remapping.xlsx')
    if 'centroid' in output:
        df = pd.DataFrame(output['centroid'])
        # df.to_csv(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/centroid_remapping.csv')
        df.to_excel(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/centroid_remapping.xlsx')
    if 'context' in output:
        for context in output['context']:
            df = pd.DataFrame(output['context'][context])
            # df.to_csv(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/context_output.csv')
            df.to_excel(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/' + str(context) + '_output.xlsx')

    print('Total run time: ' + str(time.time() - start_time))

if __name__ == '__main__':
    main()

