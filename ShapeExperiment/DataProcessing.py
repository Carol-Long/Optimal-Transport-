import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Load images
# Define path to the data directory
data_dir = Path('/Users/carollong/Desktop/OptimalTransport/ShapeProject/GroupedData')

# Categories
bunny = data_dir / 'bunny'
calf = data_dir / 'calf'
fish = data_dir / 'fish'
hand = data_dir / 'hand'
man = data_dir / 'man'
plane = data_dir / 'plane'
pokemon = data_dir / 'pokemon'
stingray = data_dir / 'stingray'
tool = data_dir / 'tool'

# Get the list of directories of all the images
all_dir = []
all_dir.append([f for f in bunny.glob("*.*")])
all_dir.append([f for f in calf.glob("*.*")])
all_dir.append([f for f in fish.glob("*.*")])
all_dir.append([f for f in hand.glob("*.*")])
all_dir.append([f for f in man.glob("*.*")])
all_dir.append([f for f in plane.glob("*.*")])
all_dir.append([f for f in pokemon.glob("*.*")])
all_dir.append([f for f in stingray.glob("*.*")])
all_dir.append([f for f in tool.glob("*.*")])

categories = ['bunny','calf','fish','hand','man','plane','pokemon','stingray','tool']

# list of all distributions, one category in a list
all_img = []

# Reshape to size kxk
k = 128

# Go through all the images. Save distributions in list
for dir in all_dir:
    temp_images = []
    for img in dir:
        tempdir = str(img)
        temp = plt.imread(tempdir, 0)
        #convert to greyscale
        if len(temp.shape)>2:
            temp = temp[:,:,2]
        temp = cv2.resize(temp, (k,k))
        #get dist
        temp = temp / np.sum(temp)
        temp_images.append(temp)
    all_img.append(temp_images)
        
#convert to np to prepare for model training
bunny_dist = np.array(all_img[0])
calf_dist = np.array(all_img[1])
fish_dist = np.array(all_img[2])
hand_dist = np.array(all_img[3])
man_dist = np.array(all_img[4])
plane_dist = np.array(all_img[5])
pokemon_dist = np.array(all_img[6])
stingray_dist = np.array(all_img[7])
tool_dist = np.array(all_img[8])

#save as .npy file
np.save('bunny.npy', bunny_dist)
np.save('calf.npy', calf_dist)
np.save('fish.npy', fish_dist)
np.save('hand.npy', hand_dist)
np.save('man.npy', man_dist)
np.save('plane.npy', plane_dist)
np.save('pokemon.npy', pokemon_dist)
np.save('stingray.npy', stingray_dist)
np.save('tool.npy', tool_dist)
