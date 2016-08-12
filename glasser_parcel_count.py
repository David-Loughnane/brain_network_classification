import scipy.io
import numpy as np

FILE_PATH = "/vol/vipdata/data/HCP100/"

parcels_source_L = scipy.io.loadmat(FILE_PATH + "thesis/group/GLASS/GLASS_PRO_PARCELs_{0}.mat".format('L'))
parcels_L = np.array(parcels_source_L['PARCELs'][0][0][0][0][0])

min_parcel = 9999
max_parcel = -10
for parcel in parcels_L:
    if parcel > max_parcel:
        max_parcel = parcel
    if parcel < min_parcel:
        min_parcel = parcel
print 'max_parcel', max_parcel
print 'min_parcel', min_parcel
