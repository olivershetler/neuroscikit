"""

UNIT MATCHER PSEUDOCODE LOGIC/SIMULATION

def Top level function (takes directory/list of directory/list of paths etc..., settings dict has settings like ppm, smoothing factor and which devices are present T/F etc...)
    
    study = make_study(directory, settings) -> loading + organizing data, flexible for any number of sessions + animals, can use this same function for testing only two sequential sessions
    study.make_animals() --> will organize sessions sequentially and by animal id

    for animal in study.animals():
        
        frst_session = None
        sequential_session = None --> keeps track of current and previous sessions to update cut values as we iterate through animals and sessions. 
        --> Otherwise would have to write a separate second for loop recomparing the feature dicts in sequential pair wise manner which means you also would have to store the 
        --> nested feature dicts in a data structure to then index into and match units. Tracking curr and prev just makes things faster. Either tracks session instance or feature dict for each sessions

        for session in animal.sessions():
            
            feature_dict = extract_features(session) --> feature dict holds all extracted spike and unit features

            if not the very first session:

                compare(first_session, sequential_session)
                match_units(first_session, sequential_session)
                write_cut(sequential_session) --> only update second sessions releative to first

            # update first_session to prev session and sequential session to new session

"""

######### Points of confusion / to be clarified #########

# extract_features() will take in a session object and extract spike features from the SpikeClusterBatch() class which is a colelction of SpikeCluster() classes 
# which is a collection of Spike(event) for one unit) ?

# functions that compute single spike features will take in as input a Spike(event) from a Spike Cluster? OR should Spike(event) have a function to return a dictionary of waveforms ? So are inputs at 
# this level our core/lib classes or not?

# extract_features() returns a dictionary of features. There is one dictionary per session? Or a list of dictionaries with one per cell? 
# Or a session dictionary with nested cell dictionaries with features for each spike inside?

# compare_sessions() simply computes the jensen shannon distance? Or collects difference distances into the aggregate divergence score?
    # compare_sessions() calls _extract_full_matches and _guess_remaining matches. These functions perform actions on the "bipartite divergence matrix"
    # How does the bipartite divergence matrix relate to the aggregate divergence score? What is the shape/dimensions of this matrix? Is it a result of computing divergence scores across extracted features?
    # larger vs smaller bipartite divergence matrix?
    # when we look to minimze the sum, we aim to use e.g. sccipy.optimize or smtg similar to find the pairings that minimize the sum?





