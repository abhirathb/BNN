import os
import sys
bnn_path=os.environ['BNNPATH']
sys.path.append(bnn_path)
import numpy as np

from network import BNN
from Sampler import HMC_sampler
from Layer import *

from pycuda import gpuarray

dim = int(sys.argv[3])

maf = sys.argv[1]
hc = sys.argv[2]
X = np.loadtxt(maf)[::,0:dim] #importing MAF for given dataset
Y = np.loadtxt(hc) #hotcoding of control/case values of dataset

N = len(X)
n_preds = X.shape[1]
print n_preds
n_hidden = 5
hidden_layer = Sigmoid_Layer(n_units=n_hidden,n_incoming=n_preds,N=N) #defining a sigmoid hidden layer with 5 nodes and capacity to take from n_preds input variables and a total of N subjects

hidden_layer_prior = ARD_Prior(shape=5.0, scale=2.0, layer=hidden_layer)
hidden_layer.setPrior(hidden_layer_prior)

n_classes=Y.shape[1] #number of classes. 

output_layer = Softmax_Layer(n_classes=n_classes,n_incoming=n_hidden,N=N)
output_layer_prior = Gaussian_Layer_Prior(shape=0.1,scale=0.1,layer=output_layer)
output_layer.setPrior(output_layer_prior)

wh = np.loadtxt('inputs/sigmoid_%d_5'%dim)
bh = np.loadtxt('inputs/sigmoid_bias_5')
ws = np.loadtxt('inputs/softmax_w_5')
bs = np.loadtxt('inputs/softmax_bias_5')



hidden_layer.setWeights(gpuarray.to_gpu(wh.copy().astype(np.float32)))
hidden_layer.setBiases(gpuarray.to_gpu(bh.reshape(1,5).copy().astype(np.float32)))
output_layer.setWeights(gpuarray.to_gpu(ws.copy().astype(np.float32)))
output_layer.setBiases(gpuarray.to_gpu(bs.reshape(1,2).copy().astype(np.float32)))


layers = []
layers.append(hidden_layer)
layers.append(output_layer)

net = BNN(X=X,Y=Y,layers=layers)  #creating the BNN object

#net.initialize(iters=2000,verbose=True,step_size=1e-5,include_prior=True) 

epsilon = float(sys.argv[4])
hmc = HMC_sampler(net,L=1,eps=epsilon,scale=True)

hmc.simple_annealing_sim(n_keep=10,n_burnin=0,eta=0.9,T0=1.0,persist=0.75,verbose=True)

