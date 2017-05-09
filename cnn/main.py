import load
import random
import matplotlib.pyplot as plt
import cnn
import numpy as np
import utils

def loss(model, iBatches, aBatches, lBatches):
    loss = 0
    for i in range(iBatches.shape[0]):
        loss += model.evaluate(iBatches[i], aBatches[i], lBatches[i])
    loss /= iBatches.shape[0]
    return loss
        
size = 256 
n = -1 
batchsize = 256 

aux_train, likes_train, images_train = load.load('cnn_data_train.csv', size=size, n=n)
aux_test, likes_test, images_test = load.load('cnn_data_test.csv', size=size, n=-1)
iBatches_test, aBatches_test, lBatches_test = \
    utils.batchify(images_test, aux_test, likes_test, batchsize, jumble=True)

model = cnn.CNN(size, target=0)
epochs = 3 
for e in range(epochs):
    iBatches_train, aBatches_train, lBatches_train = \
        utils.batchify(images_train, aux_train, likes_train, batchsize, jumble=True)
    print("Test loss at epoch", e, ":", loss(model, iBatches_test, aBatches_test, lBatches_test))
    print("Train loss at epoch", e, ":", loss(model, iBatches_train, aBatches_train, lBatches_train))
    print()
    model.train(iBatches_train, aBatches_train, lBatches_train)

print(model.get_last_layer(iBatches_test[0], aBatches_test[0]))

model.save("../models/model.ckpt")
