import os
from time import time
import scipy.io 
from scipy.sparse import csgraph
import numpy as np
import networkx as nx
import nilearn.connectome
import csv


FILE_PATH = "/vol/vipdata/data/HCP100/"
FILE_PATH_MAPPING = "/vol/medic02/users/sparisot/ClionProjects/openGm_maxflow/build/GrAverage1001/"


VOXEL_TOTAL_L = 29696
VOXEL_TOTAL_R = 29716
HEMIS_PARCELS = 41
BRAIN_PARCELS = 82
TS_LENGTH = 2400
NUM_SUBJECTS = 100
PARCELLATION = "AAL"
CORRELATION = "partial correlation"

#raw_feature_vector = np.zeros((NUM_SUBJECTS,(BRAIN_PARCELS*(BRAIN_PARCELS-1)/2)), dtype=np.float)
#dg_feature_vector = np.zeros((NUM_SUBJECTS,BRAIN_PARCELS), dtype=np.float)
#bc_feature_vector = np.zeros((NUM_SUBJECTS,BRAIN_PARCELS), dtype=np.float)
#pr_feature_vector = np.zeros((NUM_SUBJECTS,BRAIN_PARCELS), dtype=np.float)
#evc_feature_vector = np.zeros((NUM_SUBJECTS,BRAIN_PARCELS), dtype=np.float)
avg_feature_vector = np.zeros((NUM_SUBJECTS,BRAIN_PARCELS), dtype=np.float)
hks_feature_vector = np.zeros((NUM_SUBJECTS,BRAIN_PARCELS), dtype=np.float)



subjectIDs = []
with open(FILE_PATH + "subjectIDs100.txt", "r") as f:
	for line in f:
		subjectIDs.append(line.rstrip('\n'))


def functional_parcellation_mapping(subject, hemisphere, parcels, parcel_count):
	#### FUNCTIONAL TIME SERIES ####
	functional_ts_source = scipy.io.loadmat(FILE_PATH + "{0}/processed/{0}_dtseries_fix_1_normalized_corrected_{1}.mat".format(subjectIDs[subject], hemisphere))
	vxl_func_ts = np.array(functional_ts_source['dtseries1'])

	if hemisphere == "L":
		HEMIS_PARCELS = 40
	else:
		HEMIS_PARCELS = 42

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


# LEFT #
#### GLASSER PARCELLATION MAPPING ###

#parcels_source_L = scipy.io.loadmat(FILE_PATH + "thesis/group/GLASS/GLASS_PRO_PARCELs_{0}.mat".format('L'))
parcels_source_L = scipy.io.loadmat(FILE_PATH + "thesis/group/AAL/AAL_PRO_PARCELs_{0}.mat".format('L'))
parcels_L = np.array(parcels_source_L['PARCELs'][0][0][0][0][0])
parcels_L = parcels_L.astype(int)
parcels_L -= 1

#### GRAMPA PARCELLATION MAPPING ####
'''
parcel_list_L = []
with open(FILE_PATH_MAPPING + "{0}/fmri_GrAverage1001_{1}_".format('L', HEMIS_PARCELS)) as f:
	for line in f:
		for word in line.split():
			parcel_list_L.append([word])
parcels_L = np.array(parcel_list_L)
parcels_L = parcels_L.astype(int)
'''
# number of voxels assigned to each parcel to calculate average
parcel_count_list_L = []
for i in range(40): #40
    parcel_count_list_L.append([i,0])
parcel_count_L = np.array(parcel_count_list_L)

for row in parcels_L:
    parcel_count_L[row[0]][1] += 1

# GLASSER RIGHT #

#parcels_source_R = scipy.io.loadmat(FILE_PATH + "thesis/group/GLASS/GLASS_PRO_PARCELs_{0}.mat".format('R'))
parcels_source_R = scipy.io.loadmat(FILE_PATH + "thesis/group/AAL/AAL_PRO_PARCELs_{0}.mat".format('R'))
parcels_R = np.array(parcels_source_R['PARCELs'][0][0][0][0][0])
parcels_R = parcels_R.astype(int)
parcels_R -= 1

#### GRAMPA RIGHT ####
'''
parcel_list_R = []
with open(FILE_PATH_MAPPING + "{0}/fmri_GrAverage1001_{1}_".format('R', HEMIS_PARCELS)) as f:
	for line in f:
		for word in line.split():
			parcel_list_R.append([word])
parcels_R = np.array(parcel_list_R)
parcels_R = parcels_R.astype(int)
'''
# number of voxels assigned to each parcel to calculate average
parcel_count_list_R = []
for i in range(42):		#42
    parcel_count_list_R.append([i,0])
parcel_count_R = np.array(parcel_count_list_R)

for row in parcels_R:
    parcel_count_R[row[0]][1] += 1


connectome_input = []
for subject in range(len(subjectIDs)):
#for subject in range(3):
	print subject, subjectIDs[subject]
	left_parcellated_ts = functional_parcellation_mapping(subject, "L", parcels_L, parcel_count_L)
	right_parcellated_ts = functional_parcellation_mapping(subject, "R", parcels_R, parcel_count_R)
	whole_brain_ts = np.concatenate((left_parcellated_ts, right_parcellated_ts), axis=0).transpose()	
	connectome_input.append(whole_brain_ts)
#single_array = np.array(connectome_input)        
#print single_array.shape

# functional correlation matrix
conn_measure = nilearn.connectome.ConnectivityMeasure(kind=CORRELATION)
t0 = time()
func_corr = conn_measure.fit_transform(connectome_input)
t1 = time()
print("Connectome time: {} seconds".format(round(t1-t0,3)))
print 'Input vector shape: ', len(connectome_input), len(connectome_input[0]), len(connectome_input[0][0])
print 'Connectome shape: ', func_corr.shape


for subject in range(NUM_SUBJECTS):
	#for subject in range(3):
        #func_corr[subject][func_corr[subject]<0] = 0
	#print 'SUBJECT ', subject 
	time_0 = time()

	### BASELINE CORRELATIONS ###
	#raw_feature_vector[subject] = func_corr[subject][np.triu_indices(BRAIN_PARCELS,1)]

	### CONSTRUCT GRAPH ###
	#t0 = time()
	#G = nx.from_numpy_matrix(func_corr[subject])
	#t1 = time()
	#print("Graph time: {} seconds".format(round(t1-t0,3)))
	#print G.number_of_edges()
    
	### DEGREE ###
	#dg_feature_vector[subject] = nx.degree(G, weight='weight').values()

	### BETWEENESS CENTRALITY ###
	#t0 = time()
	#bc_feature_vector[subject] = nx.betweenness_centrality(G, weight='weight').values()
	#t1 = time()
	#print("BC time: {} seconds".format(round(t1-t0,3)))

	### EIGENVECTOR CENTRALITY ###
	#t0 = time()
	#evc_feature_vector[subject] = nx.eigenvector_centrality(G, weight='weight', max_iter=10000, tol=1e-05).values()
	#t1 = time()
	#print("EVC time: {} seconds".format(round(t1-t0,3)))
	
	### PAGERANK ###
	#t0 = time()
	#pr_feature_vector[subject] = nx.pagerank(G, weight='weight', max_iter=10000, tol=1e-04).values()
	#t1 = time()
	#print("PR time: {} seconds".format(round(t1-t0,3)))

	### HEAT KERNEL ###
	#t0 = time()
	degree_vector = np.sum(func_corr[subject], axis=1)
	diag_stren = np.diag(degree_vector)
	laplace = np.subtract(diag_stren, func_corr[subject])


	diag_stren[np.diag_indices(BRAIN_PARCELS)] = 1 / (degree_vector**0.5)
	norm_laplace = np.dot(diag_stren, func_corr[subject]).dot(diag_stren)

	neg_norm_laplace = np.negative(norm_laplace)
	heat_kernel = np.exp(neg_norm_laplace)


	print 'PRINT HEAT KERNEL SIG'
	HKS = heat_kernel.diagonal()
	print HKS
	print '\n'


	print 'AVG TEMP FUNC'
	AVG = np.fill_diagonal(heat_kernel, 0)
	AVG = np.sum(heat_kernel, axis=1)
	AVG = np.true_divide(AVG, BRAIN_PARCELS-1)
	print AVG

	avg_feature_vector[subject] = AVG
	hks_feature_vector[subject] = HKS

	#t1 = time()
	#print("PR time: {} seconds".format(round(t1-t0,3)))



	time_1 = time()

print("Feature Vector construction time: {} seconds".format(round(time_1-time_0,3)))
#print 'raw: ', raw_feature_vector.shape
#print 'dg: ', dg_feature_vector.shape
#print 'bc: ', bc_feature_vector.shape
#print 'evc: ', evc_feature_vector.shape
#print 'pr: ', pr_feature_vector.shape

print 'saving text'
#np.savetxt('data/partial/' + PARCELLATION + '_' + str(BRAIN_PARCELS) + '_partial_raw.txt', raw_feature_vector)
#np.savetxt('data/partial/' + PARCELLATION + '_' + str(BRAIN_PARCELS) + '_partial_dg.txt', dg_feature_vector)
#np.savetxt('data/partial/' + PARCELLATION + '_' + str(BRAIN_PARCELS) + '_partial_bc.txt', bc_feature_vector)
#np.savetxt('data/partial/' + PARCELLATION + '_' + str(BRAIN_PARCELS) + '_partial_evc.txt', evc_feature_vector)
#np.savetxt('data/partial/' + PARCELLATION + '_' + str(BRAIN_PARCELS) + '_partial_pr.txt', pr_feature_vector)
np.savetxt('data/partial/' + PARCELLATION + '_' + str(BRAIN_PARCELS) + '_partial_avg.txt', avg_feature_vector)
np.savetxt('data/partial/' + PARCELLATION + '_' + str(BRAIN_PARCELS) + '_partial_hks.txt', hks_feature_vector)
print 'text saved'


