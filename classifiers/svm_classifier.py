import sys
import os
#from memory_profiler import memory_usage
#from time import time
import math
import numpy as np
from sklearn import svm
#from sklearn import linear_model
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
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

	features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(loaded_features, labels, test_size=0.20, random_state=43)
	
	tuned_parameters = [{'C' : [0.01, 0.1, 1, 10, 50, 100 , 500 , 1000] , 'kernel': ['linear', 'rbf']}]	

	#### INSTANTIATE CLASSIFIER ####
	#logr = linear_model.LogisticRegression() 	

	clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
	clf.fit(features_train, labels_train)
	predictions = clf.predict(features_test)

	kernel_param = clf.best_params_['kernel']
	regular_param = clf.best_params_['C']

	clf2 = svm.SVC(kernel=kernel_param, C=regular_param)
	scores = cross_validation.cross_val_score(clf, loaded_features, labels, cv=5)

	print "Feature vector: ", input_features
	#print "Size of feature vector: ", loaded_features.shape
	#print clf.coef_
	#print predictions	
	print clf.best_params_
	#print("Prediciton accuracy: {:.2%}".format(metrics.accuracy_score(labels_test, predictions)))
	print "Accuracy: {0}, Std Dev: {1}".format(round(scores.mean(),3)*100, round(scores.std(),3)*100)

	#print("Memory usage after: {}MB".format(memory_usage()))


for dirName, subdirList, fileList in os.walk(NETWORKS_PATH):
	for file in fileList:
		cv_classifier(os.path.join(dirName, file))
		print '\n'

