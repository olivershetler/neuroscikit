# Outside imports
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set necessary paths / make project path = ...../neuroscikit/



PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)
lab_path = os.path.abspath(os.path.join(PROJECT_PATH, os.pardir))
print(lab_path)

# Internal imports
from _prototypes.average_spike_width.src.spike_width import extract_average_spike_widths
from _prototypes.average_spike_width.src.settings import settings_dict

# Read write modules
from x_io.rw.axona.batch_read import make_study

# Outside imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def main():
    root = tk.Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(parent=root,title='Please select a data directory.')
    output_dir = filedialog.askdirectory(parent=root,title='Please select an output directory.')
    study = make_study(input_dir,settings_dict)
    study.make_animals()
    output = extract_average_spike_widths(study)
    df = pd.DataFrame(output)
    df.to_csv(output_dir + '/spike_widths.csv')

if __name__ == '__main__':
    main()