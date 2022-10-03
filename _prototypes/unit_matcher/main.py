"""

This module reads axona cut and tetrode files.
It then extracts the spike waveforms from the cut file.
It then matches the spike waveforms to the units in the tetrode file.
It then produces a dictionary of the spike waveforms for each unit.
It then extracts features from the spike waveforms for each unit.
It then matches the spike waveforms to the units in the tetrode file.
It then produces a remapping of the units in the tetrode file.
It then applies the remapping to the cut file data.
It then writes the remapped cut file data to a new cut file.
Read, Retain, Map, Write
"""