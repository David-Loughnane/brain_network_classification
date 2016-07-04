import os
import scipy.io 
import numpy as np
import csv


VOXEL_TOTAL = 29696
VOXEL_TOTAL_R = 29716
PARCEL_TOTAL = 76 #* 2
TS_LENGTH = 2400
NUM_SUBJECTS = 100


FILE_PATH = "/vol/vipdata/data/HCP100/"
#FILE_PATH = "../data/HCP100/"

subjectIDs = []
with open(FILE_PATH + "subjectIDs100.txt", "r") as f:
	for line in f:
		subjectIDs.append(line.rstrip('\n'))

feature_vector = np.zeros((NUM_SUBJECTS,(PARCEL_TOTAL*(PARCEL_TOTAL-1)/2)), dtype=np.float)


def functional_parcellation_mapping(hemisphere):

	for subject in range(1):#range(len(subjectIDs)):

		''' DESTRIEUX PARCELLATION MAPPING '''
		parcels_source = scipy.io.loadmat(FILE_PATH + "{0}/processed/{0}_aparc_a2009s_{1}.mat".format(subjectIDs[subject], hemisphere))
		parcels = np.array(parcels_source['aparc'])


		# number of voxels assigned to each parcel
		parcel_count_list = []
		for i in range (PARCEL_TOTAL):
			parcel_count_list.append([i,0])
		parcel_count = np.array(parcel_count_list)

		for row in parcels:
			parcel_count[row[0]][1] += 1


		''' FUNCTIONAL TIME SERIES '''
		functional_ts_source = scipy.io.loadmat(FILE_PATH + "{0}/processed/{0}_dtseries_fix_1_normalized_corrected_{1}.mat".format(subjectIDs[subject], hemisphere))
		vxl_func_ts = np.array(functional_ts_source['dtseries1'])

		# add voxels BOLD ts to parcel TS
		parcel_func_ts = np.zeros((PARCEL_TOTAL, TS_LENGTH), dtype=np.float)
		for i in range(VOXEL_TOTAL):
			parcel = parcels[i]
			parcel_func_ts[parcel] += vxl_func_ts[i]

		# average parcel TS
		for i in range(PARCEL_TOTAL):
			if parcel_count[i][1] != 0:
				parcel_func_ts[i] /= parcel_count[i][1]



left_parcellated_ts = functional_parcellation_mapping("left")

# functional correlation matrix
func_corr = np.corrcoef(left_parcellated_ts)
func_corr = np.nan_to_num(func_corr)
'''
# extract upper right hand corner of correlation matrix
for i in range((PARCEL_TOTAL*(PARCEL_TOTAL-1)/2)):
	for j in range(PARCEL_TOTAL):
		for k in range(j+1, PARCEL_TOTAL):
			feature_vector[subject][j] = func_corr[j][k]
'''
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

print('FUNCTIONAL CORRELATION MATRIX: ' + str(subject))
print("min: ", min_correl)
print("max: ", max_correl)
print("zeros: ", zero_count)
print(func_corr)
print('\n')
'''

print('FUNCTIONAL CORRELATION FEATURE VECTOR')
print(type(feature_vector))
print(feature_vector.shape)
np.savetxt('functional_features_vector.txt', feature_vector)






