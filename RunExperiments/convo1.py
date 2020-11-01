import numpy as np
import ot
import matplotlib.pyplot as plt

# Load Data
training_covid = np.load('covid_train.npy')
validation_covid = np.load('covid_val.npy')

training_non = np.load('non_train.npy')
validation_non = np.load('non_val.npy')

# regularization parameter
reg = [0.0015,0.0018,0.0021]

#Write to file
f = open('Results.txt', 'a')

for i in range(len(reg)):
    examples = len(training_covid)
    weights = [1/examples for i in range(examples)]
    X = ot.bregman.convolutional_barycenter2d(training_covid, reg[i], weights)

    # regularization parameter
    examples1 = len(training_non)
    weights1 = [1/examples1 for i in range(examples1)]
    Y = ot.bregman.convolutional_barycenter2d(training_non, reg[i], weights1)
    
    nameCovid = "convoCovid" + str(i) +".csv"
    nameNon = "convoNon" + str(i) +".csv"
    # save barycenters
    np.savetxt(nameCovid, X, delimiter=",")
    np.savetxt(nameNon, Y, delimiter=",")

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
    
    #tp,fp,tn,fn
    f.write("Convolutional2D Results:\n")
    f.write("Reg: "+str(reg[i])+"\n")
    f.write("TP: % d, FP: % d, TN: % d, FN:  % d\n" %(tp, fp, tn, fn))
    f.write("Accuracy: % .2f, Recall: % .2f, Selectivity: % .2f\n" %(accuracy, recall, selectivity))

f.close()
