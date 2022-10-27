
---> Study() - WORKSPACE CLASS (HIGHLY USED)

The input to a study is a dictionary with all the sessions to be added. The key/order of sessions is irrelevant as sorting is optionally done
using the Study() class make_animals() function.

The study class is to allow different formatting/grouping of session classes. E.g. sorting sequentially, grouping by animal, etc...

This is also a top level class to allow different comparisons for example one can have a study with subjects of certain type
and another study with opposite/different requirements for subjects. Future scripts can run comparisons across sessions in studies.

Study() enables Session() and Annimal()

    ---> Session() - WORKSPACE CLASS (HIGHLY USED)

    The inputs to a session are OPTIONAL. They can be SessionData() or SessionMetadata() objects. This is to allow both top down and bottom up
    creating of the class. So core classes can be collected, sorted as data or metadata and then added to create a Session() OR a session can be
    created with no data/metaadata references and will then proceed to use make_class(Class(), input_dict) to add in the core class. I believe the batch 
    process in x_io uses the latter.

    This is the most important class and holds all the class objects (core & lib) from a single animal on a single run of an experiment. All classes
    belonging to a session have a '.session_metadata' attribute. This attribute is either passed in to the input dictionary or attempted to be read from
    the Session() class. This also allows us to easily change the session that a class belongs to by just pointing to a different session metadata instance.

    This is also generally the first class made in file reading/loading. Multiple sessions are collected before a study is made and the first class for e.g.
    a new axona tetrode file is the Session() class

        ---> SessionMetadata() - CORE CLASS (ONLY TO HOLD CLASS INSTANCE REFERENCES)

        As per the name, this holds a dictionary with references to the instances of metadata. For now there are only two but the class was made
        in anticipation of more and to help segregate any metadata/data specific attributes instead of having them all be in Session()

        {'animal': AnimalMetadata, 'devices': DevicesMetadata}

            ---> AnimalMetadata() - CORE CLASS (LOW USE/IMPORTANCE since animal id is being read from folder names)

            This holds the raw metadata related to the animal. Initially this class was necessary but now I agreed with Abid that folder names can be read as the animal ID 
            e.g. Data/Mouse_A/tetrode_files...., Data/Mouse_B/tetrode files
            Apart from the Animal ID, there is no other AnimalMetadata that is required for things to run. And currently that is being read from the folder name. 

            This class is subject to either being removed or changed so we can read necessary animal metadata from the relevant files (although animal id is not in the files but rather
            the file name and is quite inconsistently formatted across the lab)

            ---> DevicesMetadata() - CORE CLASS (ONLY TO HOLD CLASS INSTANCE REFERENCES)

            This is not the most useful class and was created in anticipation of having multiple devices. As of right now it only has two possible devices but one can imagine
            cases with multiple implants, multiple trackers or simply different implant types. 

            {'axona_led_implant': TrackerMetadata(), 'implant': ImplantMetadata()}

            This would complicate extracting metadata:

            /// session_object.session_metadata.metadata['devices'].devices_dict['axona_led_tracker'] == instance of TrackerMetadata() ///

            For this reason I gave the Session() class functions that are conceptually similar to get_spike_data, get_device_metadata, get_cell_data, get_all_metadata etc.. to avoid the user having to do this directly

            So the line above can be simplified to: 
            
            /// session_object.get_devices_metadata()['axona_led_tracker'] ///

            So this doesn't look useful in the casees of having few devices but allows for much better organization if we have more. And also the idea was to prevent against having to create a unique class for every single device and having to use different if statements to see what device metadata class we have. 

                ---> TrackerMetadata() (LOW USE/IMPORTANCE)

                Raw metadata from spatial tracking device (axona led tracker here)

                ---> ImplantMetadata() (LOW USE/IMPORTANCE)

                Raw metadata from implant (should be flexible for different types, geometry etc...)

        ---> SessionData() - WORKSPACE CLASS (ONLY TO HOLD CLASS INSTANCE REFERENCES)

        Like SessionMetadata() but for core data classes not core metadata classes

        Extremely simple class with dictionary to return instance of core class 

        e.g. {'spike_train': SpikeTrain, 'spike_cluster': SpikeClusterBatch, 'spatial_spike_train: SpatialSpikeTrain2D}

        Session class also has functions to return dictionaries holding different core session data like cell data instances or spike data instances

            --> SpikeClusterBatch - CORE CLASS (HIGHLY USED)

            Created with every Session that is made. Conceptually, this class holds a collection of unsorted spikes belonging to a variety of clusters.

            Required inputs are event times, event labels and event signal. If no event labels are needed, a workaround/fix can be added to autopopulate these labels (not implemented yet).
            For example if all event times loaded are from the same cluster/cell, all event labels can be set equal. Alternatively we can add functions to check
            whether data is sorted or not and if unsorted use SpikeClusterBatch otherwise populate multiple Cell classes since data is sorted. Personally would advocate
            for making labels if they dont exist. SpikeCluster (batch/no batch) and Cell (ensemble/not) classes are conceptually useful to distinguish between analyses
            that require separated + sorted data as opposed to just aggregated spike data. 

                --> SpikeCluster - CORE CLASS (HIGHLY USED)

                THIS WILL NOT BE CREATED AS PART OF A SESSION OR STUDY. Currently, SpikeClusterBatch precedes single label SpikeCluster classes and these are instantiated
                as part of the procedure:
                
                \\\ study.make_animals() --> sort session sequentiallly --> sort_spikes_by_cell() --> filter/map/reduce specific spike times/waveforms belonging to that label --> Spike Cluster \\\

                Can add checks to see if the data is all sorted and if so use SpikeCluster. But having batch and non batch is again because of added flexibility. E.g. if data in SpikeClusterBatch 
                is all the same cell, can still be dividied e.g. into trials based on cluster label. So its more so a class to divided neural data by label, which in our case is cell label
                but could realstically follow any other grouping logic based on other properties. E.g. giving cell types different labels and groupings all spike times + waveforms belonging
                to that cell type

            --> SpikeTrainBatch - CORE CLASS (LOW USE/IMPORTANCE)

            Unlike SpikeClusterBatch, this is not created as part of the Session class. The SpikeTrain() class is used instead. BatchSpikeTrain class is maintained
            to accomodate situations where multiple spike trains are wanted. E.g. multiple trials of spike trains belonging to the same cell. To use this class will 
            require us to check, once data is loaded or based on settings dictionary, what structure we have and whether we want the batch calss or not

            --> SpikeTrain - CORE CLASS (LOW USE/IMPORTANCE)

            Every session that is made, a SpikeTrain() object will be made with it and holds all the spike times and labels. Conceptually this is never used for sorting or clustering
            or even mapping as we can use SPikeClusterBatch. This class was preserved from earlier work and the more we move froward the more unnecessary it appears. Generally, the
            SpikeCluster class and Cell class should enable the user to do everything they need. Additionally, SpikeTrain classes are highly similar in input to the SpikeCluster class
            and hold the same event times as a Cell class except that Cell classes are sorted spike trains from the collection of all spikes

            ---> Position2D - CORE CLASS (HIGH USE)

            Holds data from pos file, assocaited with a single session and can be returned with smtg similar to session.get_spatial_data(). In future that function will return
            all present spatial classes in that session

            ---> SpatialSpikeTrain2D - WORKSPACE CLASS (HIGH USE)

            Takes in Position and SpikeTrain or Cell class. It could take in a SpikeCluster(batch or not) class but only SpikeTrain and Cell are coded in at the moment. Logically thought,
            a spatial spike train should not take in anything but a Cell because thats the only sorted data structure. We don't currently pass in a SpikeTrain obj anytime except on some test files but this needs to be
            changed as part of code review bcs we don't want to make it possible to apss in spike times that include noise. 

                ---> HaftingRateMap

                Takes in SpatialSpikeTrain2D. If occ_map and spike_map are already computed, will use them directly (has reference bcs same spatial spike train 2d)

                ---> HaftingOccupancyMap

                Takes in SpatialSpikeTrain2D (has reference to rate map and spike map same spatial spike train 2d)

                --> HaftingSpikeMap

                Takes in SpatialSpikeTrain2D (has reference to rate map and occ map same spatial spike train 2d)

    ---> Animal() - WORKSPACE CLASS (HIGHLY USED)

    Animal classes take in the same input as the study except the sessions in the dictionary are ordered in time and exclusively have the same
    session_metadata.animal_id attribute

    This enables analyses that requires session ordering and grouping. Because data in the animal class is sorted, the animal has cell workspace classes.
    These include Cell, CellEnsemble, CellPopulation. The idea is to enable flexible grouping of neural responses inside an animal. 

    E.g.

        ---> CellPopulation - WORKSPACE CLASS (LOW USE/IMPORTANCE until multiple ensemble analyses and modeling)

        collections of ensemble (maybe multi-sessions or aggregated cell data with no regard to specific ensemble it belongs to)

            ---> CellEnsemble - WORKSPACE CLASS (HIGHLY USED)

            collection of cells (maybe 1 session or multi-trials for 1 cell). Also differs from batch_space classes more conceptually than anything

                ---> Cell - WORKSPACE CLASS (HIGHLY USED)

                sorted valid spike times + waveforms. Differs from SpikeCluster conceptually but holds similar data to SpikeTrain and SpikeCluster (potential redundancy issue here)


        

