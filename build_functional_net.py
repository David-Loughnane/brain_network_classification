import scipy.io 
import numpy as np



''' DESTRIEUX PARCELLATION MAPPING '''
parcels_source_left = scipy.io.loadmat('../data/HCP100/100307/processed/100307_aparc_a2009s_L.mat')
#parcels_source_right = scipy.io.loadmat('../data/HCP100/processed/100307_aparc_a2009s_R.mat')

parcels = parcels_source_left['aparc'] #+ parcels_source_right['aparc']

print 'Voxel to parcel mapping:'
print parcels
print '\n'



''' FUNCTIONAL TIME SERIES '''
functional_ts_source_left = scipy.io.loadmat('../data/HCP100/100307/processed/100307_dtseries_fix_1_normalized_corrected_L.mat')
#functional_ts_source_right = scipy.io.loadmat('../data/HCP100/100307/processed/100307_dtseries_fix_1_normalized_corrected_R.mat')

functional_ts = functional_ts_source_left['dtseries1'] #+ functional_ts_source_right['dtseries1']

print 'Functional time series:'
print functional_ts
print '\n'

