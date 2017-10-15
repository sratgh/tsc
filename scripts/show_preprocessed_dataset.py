from matplotlib import pyplot as plt
%matplotlib inline
from glob import glob
from IPython.display import Image, display, HTML
import os

s="<table><tr>"
for idx, img in enumerate(sorted(glob("example_images/preprocessed/*.png"), key=lambda x: int(os.path.split(x)[1][:-4]))):
    #display(Image(filename=img))
    s+="<td><img src='"+img+"', height='120' width='120'>"
    if (idx+1) % 7 == 0:
        #print(idx)
        s+="</td></tr><tr>"
s+="</table>" 
display(HTML(s))