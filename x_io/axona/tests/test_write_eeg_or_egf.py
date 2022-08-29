from core.data_voltage import (
    EphysSeries
    ,EphysCollection
)

from x_io.axona.write_eeg_or_egf import write_eeg_or_egf

import numpy as np

raw_series = np.random.normal(size=1000)
data_dict = {'ephys_series': raw_series, 'units': 'mV', 'sample_rate': (1000, 'Hz')}

series = EphysSeries(data_dict)
channel_dict = {'channel_1': series}
collection = EphysCollection(channel_dict)


def test_write_eeg_or_egf():
    """
    This module is for local field potential (LFP) data. These data are measured
    """
    pass