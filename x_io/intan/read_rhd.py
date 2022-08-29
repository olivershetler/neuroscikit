"""
Intan provides a module for loading .rhd data, which is included in the load_intan_rhd_format folder in this directory.
This module is essentially just for calling that module and re-organizign the data into an EphysCollection object.
"""

from x_io.intan.load_intan_rhd_format.load_intan_rhd_format import read_rhd_data

from core.data_voltage import (
    EphysSeries
    ,EphysCollection
)

def read_rhd(rhd_file_path: str):
    """
    Read the .rhd file and return the data in an EphysCollection object.
    """
    intan_data = read_rhd_data(rhd_file_path)
    ephys_data = {}

    amplifier_sample_rate = intan_data['frequency_parameters']['amplifier_sample_rate']
    for i, series in enumerate(intan_data['amplifier_data']):
        data_dict = {'ephys_series': series, 'units': 'mV', 'sample_rate': (amplifier_sample_rate, 'Hz')}
        ephys_data["channel_{}".format(i)] = EphysSeries(data_dict)
        EphysSeries(ephys_data)
    return EphysCollection(ephys_data)