import torch
import numpy as np
dat=torch.load("./train_50pct.pt")
W = dat['weight']

samples = W[:,:5]
print("w.shape: "+str(W.shape))
DVec = torch.diagonal(W)
D = torch.diag(DVec)
print("D shape: " +str(D.shape))
G = (1/(W.shape[0]-1))*torch.matmul((W-D),torch.ones(W.shape[0])) 
print("G shape: " + str(G.shape)) 
print("head of G: " + str(G[:10]))
print("Dvec shape: " + str(DVec.shape))
Diff = np.array(DVec-G)
np.save("outlier_distances.npy",Diff)
