import numpy as np
import ot
import matplotlib.pyplot as plt
import math

# Load Data
training_covid = np.load('covid_train.npy')
validation_covid = np.load('covid_val.npy')

training_non = np.load('non_train.npy')
validation_non = np.load('non_val.npy')

# Start training parameters
# A dim*n_his (pixel dist in columns)
N = len(training_covid)
A = []
M = []
for i in range(N):
    arr = training_covid[i].flatten()
    A.append(arr)
A = np.array(A).transpose()

# dimension
k = training_covid.shape[1]
# M dim*dim (euclidean dist)
for i in range(k):
    for j in range(k):
        for m in range(k):
            for n in range(k):
                M.append(math.pow(i-m,2)+math.pow(j-n,2))
M = np.reshape(M, (k*k,k*k))

# A1 dim*n_his (pixel dist in columns)
N1 = len(training_non)
A1 = []
M1 = M
for i in range(N1):
    arr = training_non[i].flatten()
    A1.append(arr)
A1 = np.array(A1).transpose()

# covid
reg = 0.002
examples = len(training_covid)
weights = np.array([1/examples for i in range(examples)]) #weights n*1
X = ot.bregman.barycenter_sinkhorn(A, M, reg, weights)

# non-covid
examples1 = len(training_non)
weights1 = np.array([1/examples1 for i in range(examples1)]) #weights1 n*1
Y = ot.bregman.barycenter_sinkhorn(A1, M, reg, weights1)

X = np.reshape(X,(k,k))
Y = np.reshape(Y,(k,k))

np.savetxt("bregmanbarysinkhornCovid.csv", X, delimiter=",")
np.savetxt("bregmanbarysinkhornNon.csv", Y, delimiter=",")


# analyze result
tp = 0
fp = 0
tn = 0
fn = 0
# calculate frobenius dist between val image and barycenters
for img in validation_covid:
    mat1 = np.subtract(img,X)
    mat2 = np.subtract(img,Y)
    dist1 = np.linalg.norm(mat1)
    dist2 = np.linalg.norm(mat2)
    # classify correct
    if dist1<dist2:
        tp+=1
    else:
        fn+=1
for img in validation_non:
    mat1 = np.subtract(img,X)
    mat2 = np.subtract(img,Y)
    dist1 = np.linalg.norm(mat1)
    dist2 = np.linalg.norm(mat2)
    # classify correct
    if dist1>dist2:
        tn+=1
    else:
        fp+=1
accuracy = (tp+tn)/(tp+tn+fp+fn)*100
recall = tp/(tp+fn)*100
selectivity = tn/(tn+fp)*100
#Write to file
f = open('Results.txt', 'a')
#tp,fp,tn,fn
f.write("Bregmanbary Sinkhorn Results:\n")
f.write("TP: % d, FP: % d, TN: % d, FN:  % d\n" %(tp, fp, tn, fn))
f.write("Accuracy: % .2f, Recall: % .2f, Selectivity: % .2f\n" %(accuracy, recall, selectivity))
f.close()



