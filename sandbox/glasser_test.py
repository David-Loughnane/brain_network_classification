import scipy.io
import numpy as np

FILE_PATH = "/vol/vipdata/data/HCP100/thesis/group/GLASS/"

hemisphere = 'L'

parcels_source = scipy.io.loadmat(FILE_PATH + "GLASS_PRO_PARCELs_{0}.mat".format(hemisphere))
print 'file data', parcels_source

parcels = np.array(parcels_source['PARCELs'])
#print parcels
print 'mapping data', parcels[0][0][0][0][0]
print len(parcels[0][0][0][0][0])
