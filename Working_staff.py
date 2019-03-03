import numpy as np
import matplotlib as plt


import matplotlib.pyplot as p
import pandas as pd
from minisom import MiniSom
from matplotlib import  pylab
from pylab import bone,pcolor,colorbar,plot,show


row_data = pd.read_csv('hotspot dataset.csv')
# keys = row_data.keys()
# for key in keys:
#     print(key)
data = row_data.drop(columns=['Location(m) from EFD-0146','Nearest Pole Location','Pole LIS number','Date','Phase'], inplace=False)
keys = data.keys()
# print(len(keys))
data[keys]=data[keys].replace(0, np.NaN)
data.fillna(data.mean(),inplace = True)
# print(data)
X = data.iloc[:].values
# Y = row_data.iloc[:, -1].values
# print(row_data)
# print(X)

som = MiniSom(x=20, y=20, input_len=len(keys))
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)
#
bone()
pylab.pcolor(som.distance_map(), cmap=plt.cm.Reds)
colorbar()
clusters = {}
for i, x in enumerate(X):
    keys = clusters.keys()
    w = som.winner(x)
    if w in keys:
        clusters[w].append(i)
    else:
        clusters[w] = []
        clusters[w].append(i)
    plot(w[0]+0.5,w[1]+0.5,'o',
         markeredgecolor='g',markerfacecolor='None',
         markersize=10, markeredgewidth=1)
show()
numbers = []
i = 0
x_axis =[]
for cluster in clusters.keys():
    x_axis.append(i)
    i += 1
    numbers.append(len(clusters[cluster]))
    print(cluster, len(clusters[cluster]))
p.bar(x_axis,numbers)
show()

