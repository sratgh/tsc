import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

folder = "data/preprocessed_training/"
#yscale = [500, 1000, 1500, 2000]
yscale = [2000, 4000, 6000, 8000]
folder_names_classes = sorted(os.listdir(folder))
l=[]
for name in folder_names_classes:
    l.append(len(glob(folder+name+"/*.png")))


plt.figure(figsize=(100,90))

plt.xlabel('Class of traffic sign',fontsize=20)   
plt.ylabel('Number samples',fontsize=20)
plt.xticks(range(1, len(l)+1), rotation='vertical', fontsize=15)
plt.yticks(yscale, fontsize=20)
plt.bar(range(1, len(l)+1), l)
plt.axhline(np.mean(l), color='r', linestyle='dashed', linewidth=2)
print(np.mean(l))
print(l)
plt.show()