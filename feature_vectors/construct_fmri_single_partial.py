import os
import scipy.io 
import nilearn.connectome
import numpy as np
import csv


VOXEL_TOTAL_L = 29696
VOXEL_TOTAL_R = 29716
HEMIS_PARCELS = 76
BRAIN_PARCELS = HEMIS_PARCELS*2
TS_LENGTH = 2400
NUM_SUBJECTS = 100


FILE_PATH = "/vol/vipdata/data/HCP100/"
FILE_PATH_MAPPING = "/vol/medic02/users/sparisot/ClionProjects/openGm_maxflow/build/"


subjectIDs = []
with open(FILE_PATH + "subjectIDs100.txt", "r") as f:
	for line in f:
		subjectIDs.append(line.rstrip('\n'))

feature_vector = np.zeros((NUM_SUBJECTS,(BRAIN_PARCELS*(BRAIN_PARCELS-1)/2)), dtype=np.float)


def functional_parcellation_mapping(subject, hemisphere):
	#### DESTRIEUX PARCELLATION MAPPING ####
	parcels_source = scipy.io.loadmat(FILE_PATH + "{0}/processed/{0}_aparc_a2009s_{1}.mat".format(subjectIDs[subject], hemisphere))
	parcels = np.array(parcels_source['aparc'])
	#print parcels

	#### GRAMPA PARCELLATION MAPPING ####
	'''	
	parcel_list = []
	with open(FILE_PATH_MAPPING + "{0}/{1}/fmri_{0}_200_initR10.txt".format(subjectIDs[subject], hemisphere)) as f:
		for line in f:
			for word in line.split():
				parcel_list.append([round(float(word))])
	parcels = np.array(parcel_list)
	parcels = parcels.astype(int)
	'''
	# number of voxels assigned to each parcel to calculate average
	parcel_count_list = []
	for i in range (HEMIS_PARCELS):
		parcel_count_list.append([i,0])
	parcel_count = np.array(parcel_count_list)

	for row in parcels:
		parcel_count[row[0]][1] += 1

	#### FUNCTIONAL TIME SERIES ####
	functional_ts_source = scipy.io.loadmat(FILE_PATH + "{0}/processed/{0}_dtseries_fix_1_normalized_corrected_{1}.mat".format(subjectIDs[subject], hemisphere))
	vxl_func_ts = np.array(functional_ts_source['dtseries1'])

	# add voxels BOLD ts to parcel TS
	parcel_func_ts = np.zeros((HEMIS_PARCELS, TS_LENGTH), dtype=np.float)
	
	if hemisphere == "L":
		for i in range(VOXEL_TOTAL_L):
			parcel = parcels[i]
			parcel_func_ts[parcel] += vxl_func_ts[i]
	elif hemisphere == "R":
		for i in range(VOXEL_TOTAL_R):
			parcel = parcels[i]
			parcel_func_ts[parcel] += vxl_func_ts[i]

	# average parcel TS
	for i in range(HEMIS_PARCELS):
		if parcel_count[i][1] != 0:
			parcel_func_ts[i] /= parcel_count[i][1]

	return parcel_func_ts


connectome_input = []
for subject in range(len(subjectIDs)):
#for subject in range(1):
	print subjectIDs[subject]
	#if subjectIDs[subject] == '397760':	
	#	continue
	#else:
	left_parcellated_ts = functional_parcellation_mapping(subject, "L")
	right_parcellated_ts = functional_parcellation_mapping(subject, "R")
	whole_brain_ts = np.concatenate((left_parcellated_ts, right_parcellated_ts), axis=0)
	connectome_input.append(whole_brain_ts)

# functional correlation matrix
conn_measure = nilearn.connectome.ConnectivityMeasure(kind='pearson')
func_corr = conn_measure.fit_transform(connectome_input)



print 'Connectome computation complete'
print 'connectome'
print func_corr.shape

'''
for subject in range(NUM_SUBJECTS):
	for i in range((BRAIN_PARCELS*(BRAIN_PARCELS-1)/2)):
		for j in range(BRAIN_PARCELS):
			for k in range(j+1, BRAIN_PARCELS):
				feature_vector[subject][j] = func_corr[subject][j][k]
'''
for subject in range(NUM_SUBJECTS):
#for subject in range(1):
	feature_vector[subject] = func_corr[subject][np.triu_indices(BRAIN_PARCELS,1)]
	'''
	max_correl = -9999
	min_correl = 9999
	zero_count = 0
	for c in feature_vector[subject]:
		if c == 0:
			zero_count += 1
		elif c > max_correl:
			max_correl = c
		elif c < min_correl:
			min_correl = c
	'''
	print('FUNCTIONAL CORRELATION MATRIX: ' + str(subject))
	'''
	print("min: ", min_correl)
	print("max: ", max_correl)
	print("zeros: ", zero_count)
	print(func_corr[subject])
	print('\n')
	'''

print('FUNCTIONAL CORRELATION FEATURE VECTOR')
print(type(feature_vector))
print(feature_vector.shape)
np.savetxt('networks/fmri_dest_pearson.txt', feature_vector)
