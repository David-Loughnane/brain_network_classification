import sys
#from time import time
import math
#from memory_profiler import memory_usage
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
from sklearn import cross_validation
from sklearn import metrics

''' LOAD FEATURE VECTOR'''
FILE_PATH = "../network_construction/networks/"
PARCELLATION_FILE = "fmri_aal_partial.txt"

loaded_features = np.loadtxt(FILE_PATH + PARCELLATION_FILE)
print("Size of feature vector: ", loaded_features.shape)

''' GENDER LABLELS '''
subject_attribs = np.genfromtxt('../class_labels.csv', delimiter=',', dtype = str, skip_header = 1)
labels_gender = []
for row in subject_attribs:
	if row[3] == 'F':
		labels_gender.append(-1)
	else:
		labels_gender.append( 1)

#print("Memory usage before: {}MB".format(memory_usage()))

#### INSTANTIATE CLASSIFIER ####
clf = RandomForestClassifier(n_estimators=100)
#clf = AdaBoostClassifier(n_estimators=100, base_estimator=tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=100, max_depth=20))

#### CROSS VALIDATION ####
#t0 = time()
scores = cross_validation.cross_val_score(clf, loaded_features, labels_gender, cv=5)
#t1 = time()
print("Parcellation: ", PARCELLATION_FILE)
print("Scores: ", scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#print("CV time: {} seconds".format(round(t1-t0,3)))
#print("Memory usage after: {}MB".format(memory_usage()))
