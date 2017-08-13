import numpy as np
import scipy.io as sio
import pandas as pd
from scipy.stats import norm, multivariate_normal
from pylab import pcolor, show, colorbar, xticks, yticks
from sklearn import preprocessing as pre
from time import time
import matplotlib.pyplot as plt

'''
This code can be run by calling the appropriate functions. 
Mainly, show$\_$cov(label, examples) will display the covariance matrix 
for a certain label class and number of training examples. 
And the function problem3(qda) will train a QDA model if qda $==$ True 
and the LDA model otherwise. It will also display the the graph 
'''


DIRECTORY = ""

# Spam data set
data_spam = sio.loadmat(DIRECTORY)


# PROBLEM 1
data = data_spam['training_data']
data_label = np.matrix(data_spam['training_labels']).T
all_data = np.append(data, data_label, 1)
X_test = pre.normalize(data_spam['test_data'])
print(data_spam['test_data'].shape)
# shuffling data set
np.random.shuffle(all_data)

# since 5172 is not divisible by 5 we do 20% of 5170 which is 1034
validation_data = all_data[0:1034, :]

y_validation = validation_data[:,-1]
X_validation = validation_data[:, 0:-1]

training_data = all_data[1035:5172, :]

y_train = np.array(training_data[:,-1])
X_train = training_data[:, 0:-1]

y_train = y_train.ravel()

X_train_normal = pre.normalize(X_train)
X_validation_normal = pre.normalize(X_validation)
training_data_normal = pre.normalize(training_data)

def contrast_normalize(a):
    l2 = np.atleast_1d(np.linalg.norm(a, ord=2, axis=0))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis=0)

#X_train_normal = contrast_normalize(X_train)
training_data_normal = np.concatenate((X_train_normal, np.array([y_train]).T), axis=1)

# mean and covariance matrix for each digit class

# function that appends all the rows for a certain label
def class_vectors(label, examples):
	class_vector = []
	for row in training_data_normal[0:examples, :]:
		if (row[-1] == label):
			class_vector.append(row[:-1])
	return np.array(class_vector)


# function that find the covariance matrix for a given class
def covariance(label, examples):
	class_sample = class_vectors(label, examples)
	mean_vector = np.mean(class_sample, axis=0)
	'''
	sample_count = class_sample.shape[0]
	cov = np.zeros((class_sample.shape[1], class_sample.shape[1]))
	for row in class_sample:
		#print(row - mean_vector)
		value = np.outer(row - mean_vector, row - mean_vector)
		#print(value)
		cov += value
	'''
	return [np.cov(class_sample.T), mean_vector]


# function that displays the covariance plot
def show_cov (label, examples):
	cov = covariance(label, examples)[0]
	pcolor(cov)
	colorbar()
	yticks(np.arange(0.5,10.5),range(0,1))
	xticks(np.arange(0.5,10.5),range(0,1))
	show()

# function that pools the covariance for LDA and finds the mean for each class 
def pooled_cov(examples):
	means = []
	cov = []
	for i in range(10):
		covar, mean = covariance(i, examples)
		means.append(np.array(mean))
		cov.append(covar)
	return [np.mean(cov, axis=0), means]

def get_priors(examples):
	priors = []
	for i in range(10):
		class_count = float(class_vectors(i, examples).shape[0])
		priors.append(class_count / float(training_data_normal.shape[0]))
	return priors

def make_invertible(matrix, eps):
	#5e-7 for LDA
	add = np.eye(X_train.shape[1])*(eps)
	return matrix + add

def close(a,b):
	if np.isclose(a,b):
		return 0
	else:
		return 1

def test(y_predicted):
	count = 0
	for i in range(y_validation.shape[0]):
		count += close(y_validation[i], y_predicted[i])
	return count/y_predicted.shape[0]


def LDA_predict(examples, eps, kaggle):
	# group each sample
	classes = []
	for i in range(2):
		class_vect = class_vectors(i, examples)
		classes.append(class_vect)
	
	means = {}
	covs = []
	# find means and covs
	for i in range(2):
		means[i] = np.mean(classes[i], axis=0)
		covs.append(np.cov(classes[i].T))

	# find pooled covariance
	cov = np.mean(np.array(covs), axis=0)


	#cov, means = pooled_cov(examples)
	pseudo = make_invertible(cov, eps)
	pseudo_inv = np.linalg.inv(pseudo)
	#priors = get_priors(examples)

	class_v = {}
	#pre made computations
	for i in range(2):
		comp1 = -0.5* np.dot(np.dot(means[i], pseudo_inv), means[i])
		comp2 = np.dot(means[i], pseudo_inv)
		class_v[i] = [comp1, comp2]

	#make predictions
	if kaggle:

		y_predicted = []
		for X in X_test:
			probs = [class_v[i][0] + np.dot(class_v[i][1], X) for i in range(2)]
			y_predicted.append(np.argmax(np.array(probs)))		
	else:

		y_predicted = []
		for X in X_validation_normal:
			probs = [class_v[i][0] + np.dot(class_v[i][1], X) for i in range(2)]
			y_predicted.append(np.argmax(np.array(probs)))
	
	return np.array(y_predicted)

def QDA_predict(examples, eps, kaggle):
	# group each sample
	classes = []
	for i in range(2):
		class_vect = class_vectors(i, examples)
		classes.append(class_vect)
	
	means = {}
	covs = []
	# find means and covs
	for i in range(2):
		means[i] = np.mean(classes[i], axis=0)
		cov = np.linalg.inv(make_invertible(np.cov(classes[i].T), eps))
		covs.append(cov)
	
	dets = [np.sum(np.log(np.linalg.eigvals(x))) for x in covs]

	class_v = {}
	#pre made computations
	for i in range(2):
		comp1 = -0.5* np.dot(np.dot(means[i], covs[i]), means[i])
		comp2 = np.dot(means[i], covs[i])
		comp3 = -0.5 * dets[i]
		class_v[i] = [comp1, comp2, comp3]
	#make predictions
	if kaggle:

		y_predicted = []
		for X in X_test:
			probs = [class_v[i][0] + class_v[i][2] + np.dot(class_v[i][1],X) -\
				0.5* np.dot(np.dot(X, covs[i]), X) for i in range(2)]
			y_predicted.append(np.argmax(np.array(probs)))		
	else:

		y_predicted = []
		for X in X_validation_normal:
			probs = [class_v[i][0] + class_v[i][2] + np.dot(class_v[i][1] -\
				0.5* np.dot(np.dot(X, covs[i]), X), X) for i in range(2)]
			y_predicted.append(np.argmax(np.array(probs)))	
	
	return np.array(y_predicted)

def train(number, qda, eps, text):
	if (text==True):
		print(" Samples: ", number)
		print("_______________________________")
		t0 = time()
		#perc = Perceptron(n_iter=10).fit(data_train.values, y_train)	
		if (qda==False):
			model_predict = LDA_predict(number, eps, False)
		else:
			model_predict = QDA_predict(number, eps, False)
		train_time = time() - t0
		print("train time: %0.3fs" % train_time)
		#validation accuracy
		print("VALIDATION ACCURACIES:")
		t0 = time()
		score_validation = test(model_predict)
		validation_time=time() - t0
		print("test time:  %0.3fs" % validation_time)
		print("error:   %0.3f" % score_validation)
		print("_______________________________")

		return score_validation
	else:
		if (qda==False):
			model_predict = LDA_predict(number, eps, False)
		else:
			model_predict = QDA_predict(number, eps, False)
		score_validation = test(model_predict)
		return score_validation

def plot(x, y, labels):
	plt.figure()
	plt.plot(x,y)
	plt.ylim([0, 1.2])
	plt.xlabel(labels[0])
	plt.ylabel(labels[1])
	plt.show()

def problem3(qda):
	errors = {}
	x = [4137]
	for e in np.logspace(-5,2, 30):
		errors[e] = train(4137, qda, e, False)
	print(min(errors.items(), key=lambda x: x[1]) )


def kaggle():
	prediction = QDA_predict(4137, 0.072789538439831533, True)
	prediction = pd.DataFrame(prediction, dtype=int)
	prediction.index.rename('Id', inplace=True)
	prediction.to_csv('outcome_spam.csv', header =['Category'])

