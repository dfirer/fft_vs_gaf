from pyts.image import GramianAngularField
from os import listdir, path
import matplotlib.pyplot as plt
import numpy as np
import shutil
import wave
import sys
import os

base_folder = 'data/cat_dog/test/'
cat_test = path.join(base_folder, 'cats/')
dog_test = path.join(base_folder, 'dog/')

cat_filepaths = [cat_test + str(f) for f in listdir(cat_test) if path.isfile(path.join(cat_test, f))]
dog_filepaths = [dog_test + str(f) for f in listdir(dog_test) if path.isfile(path.join(dog_test, f))]

print('cat test files: ' + str(len(cat_filepaths)))
print('dog test files: ' + str(len(dog_filepaths)))

fpc = cat_filepaths[5]
fpd = dog_filepaths[10]

spf1 = wave.open(fpc, "r")
spf2 = wave.open(fpd, "r")

# Extract Raw Audio from Wav File
signal1 = spf1.readframes(-1)
signal1 = np.fromstring(signal1, "Int16")

signal2 = spf2.readframes(-1)
signal2 = np.fromstring(signal2, "Int16")

c_temp = []
for i in range(len(cat_filepaths)):
    spf = wave.open(cat_filepaths[i], "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int16")

    c_temp.append(signal)

d_temp = []
for i in range(len(dog_filepaths)):
    spf = wave.open(dog_filepaths[i], "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int16")

    d_temp.append(signal)

# resizing to smallest length
c_rsz = []
for ele in c_temp:
    c_rsz.append(ele[0: min(map(len, c_temp))])

d_rsz = []
for ele in d_temp:
    d_rsz.append(ele[0: min(map(len, d_temp))])

cat_np = np.array(c_rsz)
dog_np = np.array(d_rsz)

image_size = 50
gaf = GramianAngularField(image_size)

cat = gaf.fit_transform(cat_np)
dog = gaf.fit_transform(dog_np)

output_directories = ['output/cat_dog/test/cat/', 'output/cat_dog/test/dog/']  # mmodify to generate test output

for op in output_directories:
    if not os.path.exists(op):
        os.makedirs(op)

for i in range(len(cat)):
    start_idx = cat_filepaths[i].rindex('/') + 1
    output_name = cat_filepaths[i][start_idx:-4]

    plt.imsave(path.join(output_directories[0], output_name) + ".jpg", cat[i])

for i in range(len(dog)):
    start_idx = dog_filepaths[i].rindex('/') + 1
    output_name = dog_filepaths[i][start_idx:-4]

    plt.imsave(path.join(output_directories[1], output_name) + ".jpg", dog[i])

