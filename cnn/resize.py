import pandas as pd
from scipy import misc
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

def _resizeIm(indicies):
    for i in range(indicies[0], indicies[1] + 1):
        try:
            image = misc.imread("../data/img/" + df['Filename'].iloc[i])
            image = misc.imresize(image, (gSize, gSize))
            # check image is RGB
            if len(image.shape) != 3:
                x = True
                image_temp = np.zeros((gSize, gSize, 3))
                for i in range(3):
                    image_temp[:,:,i] = image 
                image = image_temp
            misc.imsave("../data/img_fast/" + df['Filename'].iloc[i], image)
        except OSError:
            pass
    return 0 

n = -1 
global gSize, df
df = pd.read_csv("../data/cnn_data.csv")
if (n > df.shape[0]) or (n == -1):
    n = df.shape[0] 
gSize = 256 

nthreads = mp.cpu_count() 
if (nthreads > n):
    nthreads = n
p = mp.Pool(nthreads)
step = int(n / nthreads)
indicies = []
for i in range(nthreads):
    lowI = i * step
    # last thread takes remainder of work
    if i == (nthreads - 1):
        highI = n - 1
    else:
        highI = lowI + (step - 1)
    indicies.append((lowI, highI))

#for i in range(nthreads):
    #_resizeIm(indicies[i])
p.map(_resizeIm, indicies) 
