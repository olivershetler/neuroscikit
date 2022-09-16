# Waveform Sorter

The waveform sorter is a tool that matches neurons across recording sessions using aggregated features.

## Features
- spike_width
- source_vector
- wave_form (element-wise)

## Feature Overlap and Distances
- spike_width - quantile band overlap and mean absolute difference
- source_vector - quantile overlap and cosine distance
- wave_form - quantile overlap and mean absolute difference

# Tetrode Geometry Examples

## 2x2 Axona
Each wire is 25um and each bundle of 4 wires (a tetrode) is 50x50um since the wires tend to stick in that configuration together. As in 2x2 as a square. So 4 such tetrodes are aligned in such a way that total area across would be 100 x 100 microns.

## Tetrode Array
Tetrode arrays are ostensibly arranged so that each tetrode abuts the other tetrodes, resulting in a grid. However, the wires are not always aligned in a grid. This variability makes it difficult to determine the exact geometry of the tetrode array.

In order to extract directional vectors from tetrode arrays, we approximate the geometry of the tetrode array by assuming that the tetrodes are arranged in a grid, but we do not put too much confidence in the inter-tetrode coordinates. In stead, we compute the directional vectors associated with each spike for each tetrode, and then average the intersections of the rays extending from each directional vector. Essentially, we get nC2 intersections of rays, where n is the number of tetrodes. We then average the intersections of the rays to get a single directional vector for the tetrode array.

The directional vectors for each tetrode are themselves computed by taking each subset of 3 points and finding the plane that fits the points. The normal vector of the plane at its steepest incline is the directional vector for the tetrode.