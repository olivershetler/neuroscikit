from x_io.axona.axona_utils import make_time_index_version_1

from datetime import datetime

def test_make_time_index_from_rate():
    time_index = make_time_index_from_rate('09:15:56', 10, 250)
    print(time_index)