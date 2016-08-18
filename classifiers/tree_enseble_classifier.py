import sys
import os
#from memory_profiler import memory_usage
#from time import time
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn import cross_validation
from sklearn import metrics


NETWORKS_PATH = '/homes/dl4415/Documents/project/brain_network_classification/feature_vectors/data/partial/'


def cv_classifier(input_features):
	loaded_features = np.loadtxt(input_features)

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
	#clf = AdaBoostClassifier(n_estimators=100)
	clf = GradientBoostingClassifier(n_estimators=100)

	#### CROSS VALIDATION ####
	#t0 = time()
	scores = cross_validation.cross_val_score(clf, loaded_features, labels, cv=5)
	#t1 = time()

	print "Feature vector: ", input_features
	#print "Size of feature vector: ", loaded_features.shape
	#print "Scores: ", scores
	print "Accuracy: {0}, Std Dev: {1}".format(round(scores.mean(),3)*100, round(scores.std(),3)*100)

	#print("CV time: {} seconds".format(round(t1-t0,3)))
	#print("Memory usage after: {}MB".format(memory_usage()))



for dirName, subdirList, fileList in os.walk(NETWORKS_PATH):
	for file in fileList:
		cv_classifier(os.path.join(dirName, file))
		print '\n'

