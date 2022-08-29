from x_io.intan.load_intan_rhd_format.load_intan_rhd_format import read_rhd_data
from x_io.intan.read_rhd import read_rhd


import os
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
data_dir = os.path.join(parent_dir, 'neuroscikit_test_data')
test_rhd_file_path = os.path.join(data_dir, 'intan/sampledata.rhd')

def test_read_rhd_data():
    data = read_rhd_data(test_rhd_file_path)
    for key, item in data.items():
        print('\n\n', key, '\n', item)

def test_read_rhd():
    data = read_rhd(test_rhd_file_path)
    for key, series in data.channels.items():
        print('\n\n', key, '\n', series.data)