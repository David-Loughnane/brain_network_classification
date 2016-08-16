import sys
import os
#from memory_profiler import memory_usage
#from time import time
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn import cross_validation
from sklearn import metrics


NETWORKS_PATH = '/homes/dl4415/Documents/project/brain_network_classification/network_construction/networks/'


def cv_classifier(input_features):
	loaded_features = np.loadtxt(NETWORKS_PATH + input_features)

	subject_attribs = np.genfromtxt('../class_labels.csv', delimiter=',', dtype = str, skip_header = 1)
	labels = []
	for row in subject_attribs:
		#### GENDER ####
		if row[3] == 'F':
			labels.append(-1)
		else:
			labels.append( 1)

	#print("Memory usage before: {}MB".format(memory_usage()))

	#### INSTANTIATE CLASSIFIER ####
	#clf = RandomForestClassifier(n_estimators=100)
	clf = AdaBoostClassifier(n_estimators=100)

	#### CROSS VALIDATION ####
	#t0 = time()
	scores = cross_validation.cross_val_score(clf, loaded_features, labels, cv=5)
	#t1 = time()

	print("Parcellation: ", input_features)
	print("Size of feature vector: ", loaded_features.shape)
	print("Scores: ", scores)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	#print("CV time: {} seconds".format(round(t1-t0,3)))
	#print("Memory usage after: {}MB".format(memory_usage()))


for roots, dirs, files in os.walk(NETWORKS_PATH):
	if roots == NETWORKS_PATH:
		for file in files:
			cv_classifier(file)
			print '\n'
			i =+ 1
