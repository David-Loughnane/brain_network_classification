import sys
import os
import math
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics


NETWORKS_PATH = '/homes/dl4415/Documents/project/brain_network_classification/feature_vectors/data/partial/'

orig_stdout = sys.stdout
f = file('logistic_l1_sinlge_data.txt', 'w')
sys.stdout = f


subject_attribs = np.genfromtxt('../class_labels.csv', delimiter=',', dtype = str, skip_header = 1)
labels = []
for row in subject_attribs:
	#### GENDER ####
	if row[3] == 'F':
		labels.append(-1)
	else:
		labels.append(1)


def cv_classifier(input_features1, input_features2, regular_param):
	features1 = np.loadtxt(input_features1)
        if input_features2 == 9999:
                loaded_features = features1
        else:
                features2 = np.loadtxt(input_features2)
                loaded_features = np.concatenate((features1, features2), axis=1)

	features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(loaded_features, labels, test_size=0.20, random_state=43)

	clf = linear_model.LogisticRegression(penalty='l1', C=regular_param)
	scores = cross_validation.cross_val_score(clf, loaded_features, labels, cv=5)
	non_zeros_coefs = np.count_nonzero(clf3.coef_)
	
	return (round(scores.mean(),3)*100), non_zeros_coefs

    
regular_params = [.01, .1, 1, 10, 100, 1000]
for param in regular_params:
	for dirName, subdirList, fileList in os.walk(NETWORKS_PATH):
	        for i in range(len(fileList)):
	                score, nonzeros_coefs = cv_classifier(param, os.path.join(dirName, fileList[i]), 9999)
	                print param, fileList[i], 'N/A', score, nonzeros_coefs
	                '''
	                for dirName2, subdirList2, fileList2 in os.walk(NETWORKS_PATH):
	                        for j in range(i+1, len(fileList2)):
	                                if fileList[i] == fileList2[j]:
	                                	continue
	                                score, nonzeros_coefs = cv_classifier(param, os.path.join(dirName, fileList[i]), os.path.join(dirName, fileList2[j]))
	                                print param, fileList[i], fileList2[j], score, nonzeros_coefs
	                '''

sys.stdout = orig_stdout
f.close()