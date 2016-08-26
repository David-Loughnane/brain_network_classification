import scipy.io
import numpy as np


''' DESTRIEUX PARCELLATION MAPPING '''
parcels_source_left = scipy.io.loadmat('/vol/vipdata/data/HCP100/100307/processed/100307_aparc_a2009s_L.mat')
#parcels_source_right = scipy.io.loadmat('../data/HCP100/processed/100307_aparc_a2009s_R.mat')

parcels = parcels_source_left['aparc'] #+ parcels_source_right['aparc']
parcels_array = np.array(parcels)

''' CORTICAL SURFACE MASK '''
''' IDENTIFIES VOXELS ON THE CORTICAL SURFACE, NOT MEDIAL WALL '''
surface_voxel_source_left = scipy.io.loadmat('/vol/vipdata/data/HCP100/100307/processed/100307_atlasroi_cdata_L.mat')
#surface_voxel_source_right = scipy.io.loadmat('../data/HCP100/100307/processed/100307_atlasroi_R.mat')
surface_mask = surface_voxel_source_left['cdata'] # + surface_voxel_source_right['cdata']


#limit = 9999999999
edges = []
with open('/vol/vipdata/data/HCP100/100307/diffusion/preprocessed/T1w/probtrack/L/fdt_matrix1.dot', 'r') as f:
	for line in f:
                #for _, line in zip(range(limit), f):
		edge = line.split()
		#if surface_mask[int(edge[0])-1][0] == 1 and surface_mask[int(edge[1])-1][0] == 1:
                edges.append(edge)
struct_matrix = np.array(edges)
print struct_matrix.shape
'''


struct_matrix_parcels = []
for row in struct_matrix:
	#print (parcels_array[int(row[0])-1][0], parcels_array[int(row[1])-1][0], row[2])
	struct_matrix_parcels.append((parcels_array[int(row[0])-1][0], parcels_array[int(row[1])-1][0], row[2]))

struct_np_matrix_parcels = np.array(struct_matrix_parcels)
#print struct_np_matrix_parcels

compact_matrix = []
for i in range(75):
	for j in range(75):
		compact_matrix.append([i,j,int(0)])
compact_np_matrix = np.array(compact_matrix)	

#print compact_np_matrix

for row in struct_np_matrix_parcels:
	#print (int(row[0])-1)+((int(row[1])-1)*72)
	#compact_np_matrix[(int(row[0])-1)+((int(row[1])-1)*72)][2] += int(row[2])

#print compact_np_matrix
'''
