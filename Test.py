import numpy as np
import matplotlib as plt
import pandas as pd
from minisom import MiniSom
from matplotlib import  pylab
from pylab import bone,pcolor,colorbar,plot,show


row_data = pd.read_csv('heart.csv')
X = row_data.iloc[:,:-1].values
Y = row_data.iloc[:, -1].values
# print(row_data)
# print(X)

som = MiniSom(x=50,y=50, input_len=13)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

bone()
pylab.pcolor(som.distance_map().T, cmap=plt.cm.Reds)
colorbar()
markers = ['o','s']
colors = ['r','g']
# for i, x in enumerate(X):
#     w = som.winner(x)
#     plot(w[0]+0.5,w[1]+0.5,markers[Y[i]],
#          markeredgecolor=colors[Y[i]],markerfacecolor='None',
#          markersize=10, markeredgewidth=1)
show()

