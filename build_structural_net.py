import scipy.io
import numpy as np

''' CORTICAL SURFACE MASK '''
surface_voxel_source_left = scipy.io.loadmat('../data/HCP100/100307/processed/100307_atlasroi_L.mat')
#surface_voxel_source_right = scipy.io.loadmat('../data/HCP100/100307/processed/100307_atlasroi_R.mat')

surface_mask = surface_voxel_source_left['cdata'] # + surface_voxel_source_right['cdata']


limit = 10000000 
edges = []

with open('../data/HCP100/100307/diffusion/preprocessed/T1w/probtrack/L/fdt_matrix1.dot', 'r') as f:
	#for line in f:
	for _, line in zip(range(limit), f):
		edge = line.split()
		if surface_mask[int(edge[0])-1][0] != 1.0 or surface_mask[int(edge[1])-1][0] != 1.0:
			edges.append(edge)

struct_matrix = np.array(edges)

'''
edges_masked = []

i = 0
for row in struct_matrix:
	if surface_mask[int(row[0])-1][0] != 1.0 or surface_mask[int(row[1])-1][0] != 1.0:
		edges_masked.append(row)
'''

print struct_matrix.shape
print struct_matrix




''' DIFFUSION DATA '''
'''
for _ in np.nditer(surface_mask):
	if _ != 1.0:
		i += 1
'''