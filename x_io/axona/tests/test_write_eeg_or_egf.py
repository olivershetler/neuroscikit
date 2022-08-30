from core.data_voltage import (
    EphysSeries
    ,EphysCollection
)

from x_io.axona.write_eeg_or_egf import (
    write_eeg_or_egf
    ,make_eeg_or_egf_header
)

import numpy as np

raw_series = np.random.normal(size=1000)
data_dict = {'ephys_series': raw_series, 'units': 'mV', 'sample_rate': (1000, 'Hz')}

series = EphysSeries(data_dict)
channel_dict = {'channel_1': series}
collection = EphysCollection(channel_dict)

session_metadata = {
    'trial_date': '2019-01-01'
    ,'trial_time': '12:00:00'
    ,'experimenter': 'John Doe'
    ,'comments': 'This is a test'
    }

def test_make_eeg_or_egf_header():
    """
    This module is for local field potential (LFP) data. These data are measured
    """
    header = make_eeg_or_egf_header(series, session_metadata)
    print(header)


def test_write_eeg_or_egf():
    """
    This module is for local field potential (LFP) data. These data are measured
    """
    write_eeg_or_egf(collection, session_metadata, '.', 'test')
    #write_eeg_or_egf(collection, 'egf')
    pass