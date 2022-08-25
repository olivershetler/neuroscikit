from datetime import datetime, timedelta
from random import sample

def make_hms_index_from_rate(start_time, sample_length, sample_rate):
    """
    Creates a time index for a sample of length sample_length at sample_rate
    starting at start_time.

    Output is in hours, minutes, seconds (HMS)
    """
    if type(start_time) == str:
        start_time = datetime.strptime(start_time, '%H:%M:%S')
    time_index = [start_time]
    for i in range(1,sample_length):
        time_index.append(time_index[-1] + timedelta(seconds=1/sample_rate))
    str_time_index = [time.strftime('%H:%M:%S.%f') for time in time_index]
    return str_time_index

def make_seconds_index_from_rate(sample_length, sample_rate):
    """
    Same as above but output is in seconds, start_time is automatically 0
    Can think of this as doing all times - start_time so we have 0,0.02,0.04... array etc..
    """
    start_time = 0
    dt = 1/sample_rate

    time = []

    for i in range(sample_length):
        time.append(i*dt)

    return time