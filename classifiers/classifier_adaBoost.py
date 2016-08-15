import sys
from time import time
import math

from memory_profiler import memory_usage
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn import metrics

''' LOAD FEATURE VECTOR'''
loaded_features = np.loadtxt('functional_features_vector.txt')
print("Size of feature vector: ", loaded_features.shape)

''' GENDER LABLELS '''
subject_attribs = np.genfromtxt('class_labels.csv', delimiter=',', dtype = str, skip_header = 1)

labels_gender = []
for row in subject_attribs:
	if row[3] == 'F':
		labels_gender.append(-1)
	else:
		labels_gender.append( 1)

''' TRAIN/TEST SPLIT'''
features_train, features_test, labels_train, labels_test = train_test_split(loaded_features, labels_gender, test_size=0.15, random_state=42)

print("Memory usage before: {}MB".format(memory_usage()))

print("features: ", len(features_train))

#### INSTANTIATE CLASSIFIER ####
ABclassifier = AdaBoostClassifier(n_estimators=100, base_estimator=tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=100, max_depth=20))


#### TRAIN #####
t0 = time()
ABclassifier.fit(features_train, labels_train)
t1 = time()
print("Training time: {} seconds".format(round(t1-t0,3)))

print("The 1st classifier used in the ensemble method is:")
print(ABclassifier.estimators_[0])

#### PREDICT ####
t0 = time()
predictions = ABclassifier.predict(features_test)
t1 = time()

print("Prediction time: {} seconds".format(round(t1-t0,3)))
print("Prediciton accuracy: {:.2%}".format(metrics.accuracy_score(labels_test, predictions)))


print("Memory usage after: {}MB".format(memory_usage()))
