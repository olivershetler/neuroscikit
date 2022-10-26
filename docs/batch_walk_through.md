make_study() will take in a list of paths and settings:

\\\ e.g. make_study([single_directory_path], settings_dict) \\\

First step is to group files by tetrode and animal. Once all files belonging to one session
are grouped, make_session() is called

As part of the procedure, first a new session class is created. This will auto create empty session data and metadata classes.
Then the core classes are added in one after the other starting with the metadata.

- AnimalMetadata is added in first
- Then TrackerMetadata and ImplantMetadata are added. 

(No reference to the DevicesMetadata() class is needed as it too will be automatically instantiated along with SessionData and SessioMetadata. Again the idea being flexibility 
for top down or bottom up building of these classes. So devices metadata will be detected and a reference add in DevicesMetadata so we can use functions like get_all_metadata
or get_devices_metadata inside of the session class to circumvent weird nestings that are mainly for organizing neatly. So metadata wise, even if we have these weird classes like 
DevicesMetadata, the user would only ever need to add the classes that hold the raw metadata they are collecting. This requires 3 make_class calls for Animal, Tracker and Implant 
metadata classes separately.)

Then data classes are added. Similarly, SessionData is auto created so one can go directly to creating the core data classes and they will be tracked + stored in SessionData without
explicitely having to define that class (unless one wants to, in which case they can). These classe sare:

- SpikeTrain (all spike tiems and labels including noise data are passed in here)
- SpikeClusterBatch (same inputs as above but has function to pass on only valid data/signals to the individual SpikeClusters)
- Position2D (holds position data)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

For analyses requiring sorted data/workspace classes

These classes are all made upon file loading and readng. After one has a study with multiple of these sessions, we can call
study.make_animals().

This will sort sessions in time and by animal_id. Sessions are passed into Animal(). SpikeClusterBatch populates SpikeClusters for valid cellls.
Animal() also makes CellEnsemble() which makes Cell() which hold valid data. Data inside is highly similar to SpikeCluster but conceptually differs. 

One distinction we can make is that after sorting + writing a matched.cut file, we can use the new labels on the Cell class instead of having to laod in a new study.
So SpikeCluster would differ from Cell as SpikeCluster is pre sorting labels and Cell is post sorting. So we wouldnt need to load in a whole new study simply to
view data with old vs new cut labels

------------------------------------------------------------------------------------------------------------------------------------------------------------------

For analyses involving maps.

The first step in a batch process would be to create a SpatialSpikeTrain2D class for each Cell. Maybe 'Cell' should be the map workspace class since they belong
to a cell. This also helps further delineate SpikeCluster from Cell as conceptually maps would be assocaited to a cleaned up Cell not a cluster. 

Spatial spike train ccan be used directly on any map function as is and will create Hafting maps as required. If the user wants to change things slightly on the hafting map, 
they can directly create and pass in that hafting map to the functions. So even though hafting maps can be thought of as being derivates of SpatialSpikeTrain2D, the code is
flexible to both. And will not waste time recomputing previously stored maps.

------------------------------------------------------------------------------------------------------------------------------------------------------------------