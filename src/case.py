import os
import sys
bnn_path=os.environ['BNNPATH']
sys.path.append(bnn_path)
import numpy as np

from network import BNN
from Sampler import HMC_sampler
from Layer import *

maf = sys.argv[1]
hc = sys.argv[2]
X = np.loadtxt(maf)[::,0:2] #importing MAF for given dataset
Y = np.loadtxt(hc) #hotcoding of control/case values of dataset

N = len(X)
n_preds = X.shape[1]
print n_preds
hidden_layer = Sigmoid_Layer(n_units=5,n_incoming=n_preds,N=N) #defining a sigmoid hidden layer with 5 nodes and capacity to take from n_preds input variables and a total of N subjects

hidden_layer_prior = ARD_Prior(shape=5.0, scale=2.0, layer=hidden_layer)
hidden_layer.setPrior(hidden_layer_prior)

n_classes=Y.shape[1] #number of classes. 

output_layer = Softmax_Layer(n_classes=n_classes,n_incoming=5,N=N)
output_layer_prior = Gaussian_Layer_Prior(shape=0.1,scale=0.1,layer=output_layer)
output_layer.setPrior(output_layer_prior)

print hidden_layer.getWeights().shape
print hidden_layer.getBiases().shape
print output_layer.getWeights().shape
print output_layer.getBiases().shape

layers = []
layers.append(hidden_layer)
layers.append(output_layer)

net = BNN(X=X,Y=Y,layers=layers)  #creating the BNN object
print net.getTrainAccuracy()
#net.feed_forward()
#net.updateAllGradients()
#sys.exit()
#net.initialize(iters=2000,verbose=True,step_size=1e-5,include_prior=True) 


hmc = HMC_sampler(net,L=1,eps=0.001,scale=True)

#eps = hmc.find_starting_eps(verbose=True)
#print eps
#hmc.simple_annealing_sim(n_keep=1000,n_burnin=100,eta=0.9,T0=1000.0,persist=0.75,verbose=True)
#hmc.simple_annealing_sim(n_keep=10,n_burnin=0,eta=0.9,T0=1000.0,persist=0.75,verbose=True)

#hmc.getARDSummary(useMedian=False)
#for i in range(n_preds):
#    hmc.testMeanAgainstNull(i)
#hmc.plot_debug()
#hmc.plotARD(999)
#hmc.plotARD(1000)
#hmc.plotARD(1001)
#raw_input()
