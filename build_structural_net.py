import scipy.io 


#### LOAD DESTRIEUX PARCELLATION MASK ####
parcels_source_left = scipy.io.loadmat('100307_aparc_a2009s_L.mat')
#parcels_source_right = scipy.io.loadmat('100307_aparc_a2009s_R.mat')

parcels = load_parcels_left['aparc'] #+ load_parcels_right['aparc']

#### LOAD STRUCTURAL TIME SERIES ####


#### LOAD FUNCTIONAL TIME SERIES ####
functional_ts_source_left = scipy.io.loadmat('100307_dt_series_fix_normalized_connected_L.mat')
#functional_ts_source_right = scipy.io.loadmat('100307_dt_series_fix_normalized_connected_R.mat')

functional_ts = functional_ts_source_left['dtseries'] #+ functional_ts_source_right['dtseries']
