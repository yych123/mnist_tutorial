

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata

# download and read mnist
mnist = fetch_mldata('mnist-original', data_home='./')

# 'mnist.data' is 70k x 784 array, each row represents the pixels from a 28x28=784 image
# 'mnist.target' is 70k x 1 array, each row represents the target class of the corresponding image
images = mnist.data
targets = mnist.target

# make the value of pixels from [0, 255] to [0, 1] for further process
X = mnist.data / 255.
Y = mnist.target

## print the first image of the dataset
#img1 = X[5].reshape(28, 28)
#plt.imshow(img1, cmap='gray')
#plt.show()

## print the images after simple transformation
#img2 = 1 - img1
#plt.imshow(img2, cmap='gray')
#plt.show()

#img3 = img1.transpose()
#plt.imshow(img3, cmap='gray')
#plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)


# TODO:use logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

lr_classifier=LogisticRegression()
lr_classifier.fit(X_train,Y_train)
lr_predictions=lr_classifier.predict(X_test)
lr_train_accuracy=lr_classifier.score(X_train,Y_train)
lr_test_accuracy=lr_classifier.score(X_test,Y_test)


print('LogisticRegression Training accuracy: %0.2f%%' % (lr_train_accuracy*100))
print('LogisticRegression Testing accuracy: %0.2f%%' % (lr_test_accuracy*100))



# TODO:use naive bayes
from sklearn.naive_bayes import BernoulliNB

nb_classifier=BernoulliNB()
nb_classifier.fit(X_train,Y_train)
nb_predictions=nb_classifier.predict(X_test)
nb_train_accuracy=nb_classifier.score(X_train,Y_train)
nb_test_accuracy=nb_classifier.score(X_test,Y_test)



print('Naive Bayes Training accuracy: %0.2f%%' % (nb_train_accuracy*100))
print('Naive Bayes Testing accuracy: %0.2f%%' % (nb_test_accuracy*100))


# TODO:use support vector machine
from sklearn.svm import LinearSVC

svc_classifier=LinearSVC()
svc_classifier.fit(X_train,Y_train)
svc_predictions=svc_classifier.predict(X_test)
svc_train_accuracy=svc_classifier.score(X_train,Y_train)
svc_test_accuracy=svc_classifier.score(X_test,Y_test)



print('Support Vector Machine Training accuracy: %0.2f%%' % (svc_train_accuracy*100))
print('Support Vector Machine Testing accuracy: %0.2f%%' % (svc_test_accuracy*100))


# TODO:use SVM with another group of parameters

from sklearn.svm import LinearSVC

svc_classifier=LinearSVC('l2','squared_hinge',True,0.0001,0.037,)
svc_classifier.fit(X_train,Y_train)
svc_predictions=svc_classifier.predict(X_test)
svc_train_accuracy=svc_classifier.score(X_train,Y_train)
svc_test_accuracy=svc_classifier.score(X_test,Y_test)


print('Support Vector Machine Training accuracy (Parameters Adjusted): %0.2f%%' % (svc_train_accuracy*100))
print('Support Vector Machine Testing accuracy (Parameters Adjusted): %0.2f%%' % (svc_test_accuracy*100))