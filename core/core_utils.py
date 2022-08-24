from datetime import datetime, timedelta

def make_time_index_from_rate(start_time, sample_length, sample_rate):
    """
    Creates a time index for a sample of length sample_length at sample_rate
    starting at start_time.
    """
    if type(start_time) == str:
        start_time = datetime.strptime(start_time, '%H:%M:%S')
    time_index = [start_time]
    for i in range(sample_length):
        time_index.append(time_index[-1] + timedelta(seconds=1/sample_rate))
    str_time_index = [time.strftime('%H:%M:%S.%f') for time in time_index]
    return str_time_index