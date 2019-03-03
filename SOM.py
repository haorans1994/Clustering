import numpy as np
import pylab as pl
import pandas as pd


class SOM(object):
    def __init__(self,Data,output,ITER_NUM, batch_size):

        self.X = Data
        self.out = output
        self.iteration = ITER_NUM
        self.batch_size = batch_size
        self.W = np.random.rand(Data.shape[1], output[0] * output[1])
        # print('initial weight:')
        # print(self.W)
    # to determine how much the R should be.
    # The bigger iteration is, the smaller the Radius should be

    def get_radius(self,t):
        # number t iteration
        R = min(self.out)
        return int(R - float(R) * t / self.iteration)

    def get_learnrate(self,t,r):
        #get learning rate based on radius and iteration number
        LR = np.power(np.e, -r)/(t+2)
        return LR

    def updateW(self, data, t, winner):
        R = self.get_radius(t)

        pass





row_data = pd.read_csv('heart.csv')
data = row_data.drop(columns=['target'], inplace=False)

# print

# print(data.shape[1])
Weight = SOM(data,(10,10),5,1)
