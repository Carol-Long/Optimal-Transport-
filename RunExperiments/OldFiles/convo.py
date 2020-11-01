import numpy as np
import ot
from pathlib import Path
import random
import matplotlib.pyplot as plt
import cv2

# Load images 
# Define path to the data directory
data_dir = Path('/home/xl2565/ot/Data1/')

# CT images with Covid
covid = data_dir / 'CT_COVID'

# CT images without Covid
noncovid = data_dir / 'CT_NonCOVID'

# Get the list of all the images
covid_ct = [f for f in covid.glob("*.*")]
noncovid_ct = [f for f in noncovid.glob("*.*")]

# Two empty lists. We will insert the data into this list in (img_path, label) format
covid_images = []
noncovid_images = []

# Go through all the covid cases. Save distributions in list
for img in covid_ct:
    tempdir = str(img)
    temp = plt.imread(tempdir, 0)
    #convert to greyscale
    if len(temp.shape)>2:
        temp = temp[:,:,2]
    #!! what method is it using
    temp = cv2.resize(temp, (200,200))
    #get dist
    temp = temp / np.sum(temp)
    covid_images.append(temp)

# Go through all the noncovid cases. Save distributions in list
for img in noncovid_ct:
    tempdir = str(img)
    temp = plt.imread(tempdir, 0)
    #convert to greyscale
    if len(temp.shape)>2:
        temp = temp[:,:,2]
    temp = cv2.resize(temp, (200,200))
    #get dist
    temp = temp / np.sum(temp)
    noncovid_images.append(temp)

# Shuffle lists
random.shuffle(covid_images)
random.shuffle(noncovid_images)

#convert to np to prepare for model training
covid_images = np.array(covid_images)
noncovid_images = np.array(noncovid_images)

# Split into training/validation/testing for each. (arbitrary split, roughly 4:1:2)
training_covid = covid_images[0:201]
validation_covid = covid_images[201: 249]
testing_covid = covid_images[249:349]

training_non = noncovid_images[0:226]
validation_non = noncovid_images[226:281]
testing_non = noncovid_images[281:397]

# regularization parameter
reg = 0.002
examples = len(training_covid)
weights = [1/examples for i in range(examples)]
X = ot.bregman.convolutional_barycenter2d(training_covid, reg, weights)

# regularization parameter
examples1 = len(training_non)
weights1 = [1/examples1 for i in range(examples1)]
Y = ot.bregman.convolutional_barycenter2d(training_non, reg, weights1)

np.savetxt("convoCovid.csv", X, delimiter=",")
np.savetxt("convoNon.csv", Y, delimiter=",")

