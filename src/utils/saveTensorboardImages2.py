# -*- coding: future_fstrings -*-
from __future__ import print_function, division, absolute_import
import numpy as np
import re
from matplotlib import pyplot as plt
import os
import csv
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from os import listdir
from os.path import isfile, join
import glob
import pandas


LOAD_PATH="/home/finuxs/TFM/write/imagesWriting/tensorboard/movements/"
SAVE_PATH="/home/finuxs/TFM/write/imagesWriting/tensorboard/movements/"
save = 'discussion.png'
files = [f for f in sorted(glob.iglob(LOAD_PATH + "*.csv")) if isfile(join(LOAD_PATH, f))]
frames = np.zeros((len(files), 4))
loss = np.zeros((len(files), 4))
print(files)
cnt=0

for i in range(len(files)):
    idx = 0
    data = pandas.read_csv(os.path.join(LOAD_PATH, files[i]))
    media = data.values
    print(files[i])
    data = data['Discussion']
    data = data.values
    frames[i,:] = [2,4,8,10]
    loss[i,:] = [data[1],data[3],data[7],data[9]]
    plt.plot(frames[i,:], loss[i,:])
    #print('{0:.2f}'.format(loss[i,1]))
    #print('{0:.2f}'.format(loss[i,3]))
    #print('{0:.2f}'.format(loss[i,7]))
    #print('{0:.2f}'.format(loss[i,9]))

#plt.legend(['hidden GRU', 'hidden LSTM', 'repeated GRU', 'repeated LSTM'], loc='upper right')
#plt.legend(['hidden No Residual', 'hidden Residual', 'repeated No Residual', 'repeated Residual'], loc='upper right')
#plt.legend(['erd repeated GRU Residual', 'simple residual repeated LSTM', 'martinez repeated GRU', 'srivastava repeated LSTM'], loc='upper right')
#plt.legend(["even","none","window 2","window 3"], loc="upper right")
plt.legend(['ERD Residual','ERD','Martinez','Simple Residual','Srivastava'])
plt.title('Loss per Predicted Frames')
plt.xlabel('Predicted Frames')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,save))
plt.draw()
plt.close()

