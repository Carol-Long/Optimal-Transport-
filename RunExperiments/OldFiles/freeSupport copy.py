import numpy as np
import ot
import matplotlib.pyplot as plt
import math

# Load Data
training_covid = np.load('covid_train.npy')
validation_covid = np.load('covid_val.npy')

training_non = np.load('non_train.npy')
validation_non = np.load('non_val.npy')

# barycenters for covid
N = len(training_covid)
measures_locations = []
measures_weights = []
mylist = []
# dimension
k = training_covid.shape[1]
for i in range(k):
    for j in range(k):
        mylist.append([i,j])
for i in range(N):
    measures_locations.append(np.array(mylist))

for i in range(N):
    measures_weights.append(np.array(training_covid[i].flatten()))

num = 1000  # number of Diracs of the barycenter, number of pixels
d = 2
X_init = np.random.normal(0., 1., (num, d))  # initial Dirac locations
b = np.ones((num,)) / num # weights of the barycenter (it will not be optimized, only the locations are optimized)
X = ot.lp.free_support_barycenter(measures_locations, measures_weights, X_init, b, numItermax = 10000)

# barycenters for noncovid
N1 = len(training_non)
measures_locations1 = []
measures_weights1 = []
for i in range(N1):
    measures_locations1.append(np.array(mylist))
for i in range(N1):
    measures_weights1.append(np.array(training_non[i].flatten()))
Y_init = np.random.normal(0., 1., (num, d))  # initial Dirac locations
Y = ot.lp.free_support_barycenter(measures_locations1, measures_weights1, Y_init, b, numItermax = 10000)

np.save('freeSupport_covid.npy', X)
np.save('freeSupport_non.npy', Y)
#
#Xmat = np.zeros((k,k))
#w = 1/num
#for i in range(num):
#    temp = X[i]
#    xIndex = temp[0]
#    yIndex = temp[1]
#    Xmat[xIndex][yIndex] = w
#Ymat = np.zeros((k,k))
#for i in range(num):
#    temp = Y[i]
#    xIndex = temp[0]
#    yIndex = temp[1]
#    Ymat[xIndex][yIndex] = w
#
## save output
#np.savetxt("freeSupportOutput.csv", Xmat, delimiter=",")
#np.savetxt("freeSupportOutput.csv", Ymat, delimiter=",")
#
## analyze result
#tp = 0
#fp = 0
#tn = 0
#fn = 0
## calculate frobenius dist between val image and barycenters
#for img in validation_covid:
#    mat1 = np.subtract(img,Xmat)
#    mat2 = np.subtract(img,Ymat)
#    dist1 = np.linalg.norm(mat1)
#    dist2 = np.linalg.norm(mat2)
#    # classify correct
#    if dist1<dist2:
#        tp+=1
#    else:
#        fn+=1
#for img in validation_non:
#    mat1 = np.subtract(img,Xmat)
#    mat2 = np.subtract(img,Ymat)
#    dist1 = np.linalg.norm(mat1)
#    dist2 = np.linalg.norm(mat2)
#    # classify correct
#    if dist1>dist2:
#        tn+=1
#    else:
#        fp+=1
#accuracy = (tp+tn)/(tp+tn+fp+fn)*100
#recall = tp/(tp+fn)*100
#selectivity = tn/(tn+fp)*100
##Write to file
#f = open('Results.txt', 'a')
##tp,fp,tn,fn
#f.write("Free Support Results:\n")
#f.write("TP: % d, FP: % d, TN: % d, FN:  % d\n" %(tp, fp, tn, fn))
#f.write("Accuracy: % .2f, Recall: % .2f, Selectivity: % .2f\n" %(accuracy, recall, selectivity))
#f.close()
