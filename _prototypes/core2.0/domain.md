
Uniform Numerical Data Types:

- RetularTensor (AtlasTensor (np.ndarray, shape_meaning tuple), FieldTensor (np.ndarray, shape_meaning tuple))
  - assert AtlasTensor.shape == FieldTensor.shape[:len(AtlasTensor.shape)+1]
  - assert AtlasTensor.shape_meaning == FieldTensor.shape_meaning[:len(AtlasTensor.shape_meaning)+1]
  - EXAMPLES:
  - TimeSeries
    - AbstractTimeSeries (regularly spaced time series)
      - PositionTimeSeries
      - VelocityTimeSeries
      - SpeedTimeSeries
      - AccelerationTimeSeries
      - BinarySpikeTrain
      - EEGSeries
      - LFPSeries
      - RawEphysSeries
      - SpectralSeries
      - IndexTimeSeries
      - WaveformTimeSeries
  - AbstractField
    - RectangularField
      - 2DField
      - 3DField
    - DiskField
    - ManifoldField
      - 1DManifoldField
        - RingField
      - 2DManifoldField
        - TouroidField
        - SphereField
    - FieldOnModel
      - 2DFieldOnModel
        - 2DArena
      - 3DFieldOnModel
        - 3DPlayGround

- IrregularTensor (points & values, shape_meaning)
    - AbstractEventSeries (irregularly spaced events)
      - SpikeTrain
      - SpatialSpikeTrain
      - WaveformSpikeTrain
      - BehavioralEvents (markers for behavioral events)
      - StimulusEvents (markers for stimulus presentation)

Mixed Data Types:

- EntityMetadata
    - AbstractSubject (subject_id)
      - Animal (subject_id, species, genotype...)
      - Person (subject_id, species=human, name, phone, email...)
        - Patient
    - AbstractDevice (device_id)
      - ElectrodeArray (device_id, electrode_ids, array_geometry)

