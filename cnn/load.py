from scipy import misc
import numpy as np
import pandas as pd

def _loadIm(df, i, size):
    try:
        image = misc.imread("../data/img_fast/" + df['filename'].iloc[i])
        # check image is RGB
        if len(image.shape) != 3:
            image = misc.imresize(image, (size, size))
            image_temp = np.zeros((size, size, 3))
            for i in range(3):
                image_temp[:,:,i] = image 
            image = image_temp
        aux = np.zeros(3)
        aux[0] = df['numberFollowers'].iloc[i]
        aux[1] = df['numberFollowing'].iloc[i]
        aux[2] = df['numberPosts'].iloc[i]
        likes = df['numberLikes'].iloc[i]
        
    except OSError:
        image = np.zeros((size, size, 3))
        aux = np.zeros(3)
        likes = 0
        
    return aux, likes, image 

def load(filename, size=256, n=-1):
    df = pd.read_csv("../data/" + filename)
    if (n > df.shape[0]) or (n == -1):
        n = df.shape[0] 
    aux = np.zeros((n, 3))
    likes = np.zeros((n, 1))
    images = np.zeros((n, size, size, 3))

    for i in range(n):
        aux[i], likes[i], images[i] = _loadIm(df, i, size)
    return aux, likes, images 
