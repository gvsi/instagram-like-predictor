import numpy as np

def make_batch(X, Xf, y, size, start):
    if (start + size) > X.shape[0]:
        start = X.shape[0] - size
    X_batch = X[start:start + size]
    Xf_batch = Xf[start:start + size]
    y_batch = y[start:start + size]
    return X_batch, Xf_batch, y_batch

# creates n batches out of X and y
def batchify(X, Xf, y, batchsize, jumble=True):
    if jumble:
        xFold, xfFold, yFold = split(X, Xf, y, 1)
        X = xFold[0]
        Xf = xfFold[0]
        y = yFold[0]
    n = int(X.shape[0] / batchsize)
    # set up batch memory
    xBatches = np.zeros((n, batchsize, X.shape[1], X.shape[2], X.shape[3]), dtype=np.float32)
    xfBatches = np.zeros((n, batchsize, Xf.shape[1]), dtype=np.float32)
    yBatches = np.zeros((n, batchsize, y.shape[1]), dtype=np.float32)
    for i in range(n):
        xBatches[i], xfBatches[i], yBatches[i] = make_batch(X, Xf, y, batchsize, i * batchsize)
    return xBatches, xfBatches, yBatches

def split(x, xf, y, k):
    # get number of samples, n
    n = x.shape[0]
    if (n == y.shape[0] == xf.shape[0]) and (k <= n):
        # set up memory for k splits
        xFolds = np.zeros((k, int(n / k), x.shape[1], x.shape[2], x.shape[3]), dtype=np.float32)
        xfFolds = np.zeros((k, int(n / k), xf.shape[1]), dtype=np.float32)
        yFolds = np.zeros((k, int(n / k), y.shape[1]), dtype=np.float32)
        # vector holding the index of the ith unchosen sample
        freeSample = np.zeros(n, dtype=np.int32)
        for i in range(n):
            freeSample[i] = i
        # get n random numbers 
        rands = n * np.random.rand(n)
        # construct the ith of k folds 
        for i in range(k):
            for j in range(int(n / k)):
                # total iteration count over all folds
                t = int(n / k) * i + j  
                # get index of next free sample
                free = int(rands[t]) % (n - t)
                index = freeSample[free]
                # this index is taken, update free sample list
                freeSample[free: n - t - 1] = freeSample[free + 1: n - t] 
                # insert this sample into the ith fold
                xFolds[i,j] = x[index]
                xfFolds[i, j] = xf[index]
                yFolds[i,j] = y[index]
        return xFolds, xfFolds, yFolds

