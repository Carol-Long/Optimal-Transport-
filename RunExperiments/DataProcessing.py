import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Load images
# Define path to the data directory
data_dir = Path('/Users/carollong/Desktop/OptimalTransport/Data/GroupedData')

# CT images with Covid
covid = data_dir / 'CT_COVID'
covid_train = covid / 'CovidTrain'
covid_val = covid / 'CovidVal'
covid_test = covid / 'CovidTest'

# CT images without Covid
noncovid = data_dir / 'CT_NonCOVID'
non_train = noncovid / 'NonTrain'
non_val = noncovid / 'NonVal'
non_test = noncovid / 'NonTest'

# Get the list of all the images
covid_ct_train = [f for f in covid_train.glob("*.*")]
covid_ct_val = [f for f in covid_val.glob("*.*")]
covid_ct_test = [f for f in covid_test.glob("*.*")]
noncovid_ct_train = [f for f in non_train.glob("*.*")]
noncovid_ct_val = [f for f in non_val.glob("*.*")]
noncovid_ct_test = [f for f in non_test.glob("*.*")]

# Two empty lists. We will insert the data into this list
covid_images_train = []
covid_images_val = []
covid_images_test = []
noncovid_images_train = []
noncovid_images_val = []
noncovid_images_test = []

# Reshape to size kxk
k = 50

# Go through all the images. Save distributions in list
for img in covid_ct_train:
    tempdir = str(img)
    temp = plt.imread(tempdir, 0)
    #convert to greyscale
    if len(temp.shape)>2:
        temp = temp[:,:,2]
    temp = cv2.resize(temp, (k,k))
    #get dist
    temp = temp / np.sum(temp)
    covid_images_train.append(temp)

for img in covid_ct_val:
    tempdir = str(img)
    temp = plt.imread(tempdir, 0)
    #convert to greyscale
    if len(temp.shape)>2:
        temp = temp[:,:,2]
    #!! what method is it using
    temp = cv2.resize(temp, (k,k))
    #get dist
    temp = temp / np.sum(temp)
    covid_images_val.append(temp)

for img in covid_ct_test:
    tempdir = str(img)
    temp = plt.imread(tempdir, 0)
    #convert to greyscale
    if len(temp.shape)>2:
        temp = temp[:,:,2]
    #!! what method is it using
    temp = cv2.resize(temp, (k,k))
    #get dist
    temp = temp / np.sum(temp)
    covid_images_test.append(temp)

# Go through all the images. Save distributions in list
for img in noncovid_ct_train:
    tempdir = str(img)
    temp = plt.imread(tempdir, 0)
    #convert to greyscale
    if len(temp.shape)>2:
        temp = temp[:,:,2]
    temp = cv2.resize(temp, (k,k))
    #get dist
    temp = temp / np.sum(temp)
    noncovid_images_train.append(temp)
    
for img in noncovid_ct_val:
    tempdir = str(img)
    temp = plt.imread(tempdir, 0)
    #convert to greyscale
    if len(temp.shape)>2:
        temp = temp[:,:,2]
    temp = cv2.resize(temp, (k,k))
    #get dist
    temp = temp / np.sum(temp)
    noncovid_images_val.append(temp)

for img in noncovid_ct_test:
    tempdir = str(img)
    temp = plt.imread(tempdir, 0)
    #convert to greyscale
    if len(temp.shape)>2:
        temp = temp[:,:,2]
    temp = cv2.resize(temp, (k,k))
    #get dist
    temp = temp / np.sum(temp)
    noncovid_images_test.append(temp)


#convert to np to prepare for model training
covid_images_train = np.array(covid_images_train)
covid_images_val = np.array(covid_images_val)
covid_images_test = np.array(covid_images_test)
noncovid_images_train = np.array(noncovid_images_train)
noncovid_images_val = np.array(noncovid_images_val)
noncovid_images_test = np.array(noncovid_images_test)

#save as .npy file
np.save('covid_train.npy', covid_images_train)
np.save('covid_val.npy', covid_images_val)
np.save('covid_test.npy', covid_images_test)
np.save('non_train.npy', noncovid_images_train)
np.save('non_val.npy', noncovid_images_val)
np.save('non_test.npy', noncovid_images_test)
