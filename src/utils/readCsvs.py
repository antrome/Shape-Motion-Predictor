import os
from pyexcel_ods import get_data
import csv
import re

LOAD_PATH="/home/finuxs/TFM//"
SAVE_PATH="/home/finuxs/TFM/write/imagesWriting/tensorboard/movements/"
data = get_data(os.path.join(LOAD_PATH,"poses.ods"))
ll=[]
l=[]
cnt=0
gg = []
g = []

"""
for z in range(0,len(data.keys())):
    print(data.keys()[z])
    for i in range(0,len(data[data.keys()[z]])):
        if data[data.keys()[z]][i]:
            cnt +=1
            l.append(data[data.keys()[z]][i][0])
            if cnt==9:
                cnt = 0
                ll.append(l)
                l = []

    for i in range(0,len(ll)):
        g=[]
        for j in range(len(ll[i])):
            s = re.split("\[| |]", ll[i][j])
            s = [float(m) for m in s if m]
            for k in range(len(s)):
                g.append(s[k])

        gg.append(g)
    with open(os.path.join(SAVE_PATH, data.keys()[z]+'.csv'), 'w') as f:
        writer = csv.writer(f)
        for i in range(0,len(gg)):
            writer.writerow(gg[i])
"""

for i in range(0, len(data["zero vel"])):
    if data["zero vel"][i]:
        cnt += 1
        l.append(data["zero vel"][i][0])
        if cnt == 9:
            cnt = 0
            ll.append(l)
            l = []

for i in range(0, len(ll)):
    g = []
    for j in range(len(ll[i])):
        s = re.split("\[| |]", ll[i][j])
        s = [float(m) for m in s if m]
        for k in range(len(s)):
            g.append(s[k])

    gg.append(g)
with open(os.path.join(SAVE_PATH, "zero vel" + '.csv'), 'w') as f:
    writer = csv.writer(f)
    for i in range(0, len(gg)):
        writer.writerow(gg[i])
