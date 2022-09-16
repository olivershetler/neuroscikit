
def waveform_overlap(waveform_template_bands_1, waveform_template_bands_2):
    """
    Calculate the similarity between two waveforms
    """
    lower_1, upper_1 = waveform_quantile_bands_1
    lower_2, upper_2 = waveform_quantile_bands_2

    assert len(mean_1) == len(mean_2) == len(lower_1) == len(lower_2) == len(upper_1) == len(upper_2), f"Waveforms must be the same length,. The lengths are\n\nmean_1: {len(mean_1)}\nmean_2: {len(mean_2)}\nlower_1: {len(lower_1)}\nlower_2: {len(lower_2)}\nupper_1: {len(upper_1)}\nupper_2: {len(upper_2)}"

    overlap_proportions = map(_overlap_proportion, lower_1, upper_1, lower_2, upper_2)

    total_overlap_proportion = sum(overlap_proportions)/len(overlap_proportions)

    return total_overlap_proportion

def _overlap_proportion(lower_1_i, upper_1_i, lower_2_i, upper_2_i):
    """
    Find the interval between two waveforms
    """
    return _interval_overlap_proportion([lower_1_i, upper_1_i], [lower_2_i, upper_2_i])

def _interval_overlap_proportion(interval_a_i, interval_b_i):
    """
    Find the overlap between two intervals
    """
    lower_quantile_a, upper_quantile_a = interval_a_i
    lower_quantile_b, upper_quantile_b = interval_b_i

    union_size = abs(max(upper_quantile_a, upper_quantileb) - min(lower_quantile_a, lower_quantile_b))

    intersection_size = abs(min(upper_quantile_a, upper_quantileb) - max(lower_quantile_a, lower_quantile_b))

    return intersection_size / union_size

