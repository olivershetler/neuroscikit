from x_io.intan.load_intan_rhd_format.load_intan_rhd_format import read_rhd_data


import os
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
data_dir = os.path.join(parent_dir, 'neuroscikit_test_data')
test_rhd_file_path = os.path.join(data_dir, 'intan/sampledata.rhd')

def test_read_rhd_data():
    read_rhd_data(test_rhd_file_path)

