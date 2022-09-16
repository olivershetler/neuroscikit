
def waveform_overlap(waveform_bands_1, waveform_bands_2):
    """
    Calculate the similarity between two waveforms
    Waveform bands are given in the form [lower, upper] where
    lower and upper are respectively structured as [channel, channel, ...]
    """

    lower_1, upper_1 = waveform_bands_1
    lower_2, upper_2 = waveform_bands_2

    assert len(lower_1) == len(lower_2) == len(upper_1) == len(upper_2), f"Waveforms must be the same length,. The lengths are\n\nlower_1: {len(lower_1)}\nlower_2: {len(lower_2)}\nupper_1: {len(upper_1)}\nupper_2: {len(upper_2)}"

    overlap_proportions = list(map(_overlap_proportion, lower_1, upper_1, lower_2, upper_2))


    channel_overlap_proportions = list(map(lambda x: sum(x)/len(x), overlap_proportions))

    total_overlap_proportion = sum(channel_overlap_proportions)/len(channel_overlap_proportions)

    return total_overlap_proportion


def _overlap_proportion(lower_1_i, upper_1_i, lower_2_i, upper_2_i):
    """
    Find the interval between two waveforms
    """
    assert len(lower_1_i) == len(lower_2_i) == len(upper_1_i) == len(upper_2_i), f"Waveforms must be the same length,. The lengths are\n\nlower_1_i: {len(lower_1_i)}\nlower_2_i: {len(lower_2_i)}\nupper_1_i: {len(upper_1_i)}\nupper_2_i: {len(upper_2_i)}"

    interval_a_i = list(zip(lower_1_i, upper_1_i))
    interval_b_i = list(zip(lower_2_i, upper_2_i))


    overlap_proportions = list(map(_interval_overlap_proportion, interval_a_i, interval_b_i))

    return overlap_proportions


def _interval_overlap_proportion(interval_a_ij, interval_b_ij):
    """
    Find the overlap between two intervals
    """

    lower_a_ij, upper_a_ij = interval_a_ij
    lower_b_ij, upper_b_ij = interval_b_ij

    union_size = abs(max(upper_a_ij, upper_b_ij) - min(lower_a_ij, lower_b_ij))

    intersection_size = abs(min(upper_a_ij, upper_b_ij) - max(lower_a_ij, lower_b_ij))

    overlap_proportion = intersection_size/union_size

    return overlap_proportion


