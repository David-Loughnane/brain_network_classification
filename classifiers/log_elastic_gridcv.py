import sys
import os
#from memory_profiler import memory_usage
#from time import time
import math
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn import metrics


orig_stdout = sys.stdout
f = file('results/elastic_net_results_test.txt', 'w')
sys.stdout = f


NETWORKS_PATH = '/homes/dl4415/Documents/project/brain_network_classification/feature_vectors/data/partial/'


def cv_classifier(input_features1, input_features2):
	features1 = np.loadtxt(input_features1)
        if input_features2 == 9999:
                loaded_features = features1
        else:
                features2 = np.loadtxt(input_features2)
                loaded_features = np.concatenate((features1, features2), axis=1)

	subject_attribs = np.genfromtxt('../class_labels.csv', delimiter=',', dtype = str, skip_header = 1)
	labels = []
	for row in subject_attribs:
		#### GENDER ####
		if row[3] == 'F':
			labels.append(-1)
		else:
			labels.append( 1)

	features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(loaded_features, labels, test_size=0.20, random_state=43)
	#print features_train.shape, features_test.shape

	#print("Memory usage before: {}MB".format(memory_usage()))	
	tuned_parameters = [{'l1_ratio' : [.1, .3, .4, .5, .6, .7, .8, .9, 1]}]	

	#### INSTANTIATE CLASSIFIER ####
	clf = GridSearchCV(linear_model.SGDClassifier(loss='log', penalty='elasticnet'), tuned_parameters, cv=5)
        #clf = linear_model.SGDClassifier(loss='log', penalty='elasticnet')
	clf.fit(features_train, labels_train)
	#print clf.predict(features_test)

	regular_param = clf.best_params_['l1_ratio']
        #print regular_param

	clf2 = linear_model.SGDClassifier(loss='log',penalty='elasticnet', l1_ratio=regular_param)
	scores = cross_validation.cross_val_score(clf2, loaded_features, labels, cv=5)
        #print scores

	#print("Memory usage after: {}MB".format(memory_usage()))

	#clf3 = linear_model.LogisticRegression(penalty=penalty_param, C=regular_param)
	#clf3.fit(loaded_features, labels)
	#print clf3.coef_    
	
        return (round(scores.mean(),3)*100), clf.best_params_['l1_ratio']

    


for dirName, subdirList, fileList in os.walk(NETWORKS_PATH):
	#for i in range(1):
        #if dirName[-7:] == 'archive':
        #        continue
        for i in range(len(fileList)):
                score = cv_classifier(os.path.join(dirName, fileList[i]), 9999)
                print fileList[i], 'N/A', score
                for dirName2, subdirList2, fileList2 in os.walk(NETWORKS_PATH):
                        #if dirName2[-7:] == 'archive':
                        #        continue
                        for j in range(i+1, len(fileList2)):
                                if fileList[i] == fileList2[j]:
                                	continue
                                score = cv_classifier(os.path.join(dirName, fileList[i]), os.path.join(dirName, fileList2[j]))
                                print fileList[i], fileList2[j], score


sys.stdout = orig_stdout
f.close()
