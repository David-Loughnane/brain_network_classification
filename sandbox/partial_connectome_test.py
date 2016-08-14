# Fetch dataset
import nilearn.datasets
atlas = nilearn.datasets.fetch_atlas_msdl()
dataset = nilearn.datasets.fetch_adhd(n_subjects=30)


import nilearn.input_data
masker = nilearn.input_data.NiftiMapsMasker(
    atlas.maps, resampling_target="maps", detrend=True,
    low_pass=None, high_pass=None, t_r=2.5, standardize=False,
    memory='nilearn_cache', memory_level=1)
subjects = []
sites = []
adhds = []
for func_file, phenotypic in zip(dataset.func, dataset.phenotypic):
    # keep only 3 sites, to save computation time
    if phenotypic['site'] in [b'"NYU"', b'"OHSU"', b'"NeuroImage"']:
        time_series = masker.fit_transform(func_file)
        subjects.append(time_series)
        sites.append(phenotypic['site'])
        adhds.append(phenotypic['adhd'])  # ADHD/control label


import nilearn.connectome

print 'subjects'
print len(subjects)
print subjects[0].shape
print subjects

conn_measure = nilearn.connectome.ConnectivityMeasure(kind='partial correlation')
partial_matrix = conn_measure.fit_transform(subjects)

print 'connectome'
print partial_matrix.shape
print partial_matrix
