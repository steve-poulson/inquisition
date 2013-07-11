import matplotlib.pyplot as plt
from scipy.linalg import norm, eig, svd, pinv
from scipy.sparse.linalg import svds, eigs
from numpy import *
import time
import networkx as nx
#import scipy.sparse.linalg.eigen.arpack.speigs as speig
from scipy.sparse.linalg import dsolve
from operator import itemgetter
from matplotlib.colors import ListedColormap

from scipy.cluster.vq import *
from scipy.sparse import bsr_matrix,spdiags,csc_matrix,lil_matrix
from scipy.io import *
import numpy.random as r
import numpy as np
import scipy.optimize as o
from networkx import *

def PRINCOMP(A):
     [latent,coeff] = linalg.eig(cov(M)) # attention:not always sorted
     score = dot(coeff.T,M) # projection of the data in the new space
     return coeff,score,latent

def GLFM(Xmat, A, q, NumOfIter):
    """    
    %Xmat: content matrix, each row is an instance
    %A: link matrix
    %q: the dim of latent space
    %U,V: new feature matrix
    """

    betaa = 2*10^(0);
    gamma = 2*10^(0);
    Ta = 10^(6);
    
    
    nSmp = size(Xmat,1);
    I = speye(nSmp);
    
    
    for i in range(nSmp):
        A[i,i] = 0
    
    A = sparse(A);
    
    
    Z = A>0;
    Z = sparse(Z);
    
    %initialize by PCA
    
    Xmat = full(Xmat);
    
    [COEFF, X] = PRINCOMP(Xmat);
    U = X[:,1:q];
    V = X[:,1:q];
    clear X;
    
    
    
    
    Sigma = I;
    
    Sigma = (Sigma+Sigma.T)/2;
    
    
    Sigma = Sigma/betaa;
    
    Sigma1 = Sigma;
    Sigma2 = Sigma;
    
    I_q = speye(q);
    
    muu = 0;

    for t in range(NumOfIter):
        for i in range(nSmp):          
                S1 = zeros(1,nSmp);
                S2 = zeros(nSmp,1);
                Ind1 = find(Z[i,:]==1);
                Ind2 = find(Z[:,i]==1);
                X1 = U[i,:]*(U[Ind1,:]+V[Ind1,:]).T + muu;
                S1[Ind1] = exp(X1/2)./(exp(-X1/2)+exp(X1/2));
                X2 = U[Ind2,:) * (U[i,:]+V[i,:]).T + muu;
                S2(Ind2) = exp(X2/2)./(exp(-X2/2)+exp(X2/2));             
                
                Gradient = (A[i,:] + A[:,i].T - S1 - S2.T - Sigma1[i,:])*U + (A[i,:]-S1)*V;
                
                U1 = U[Ind1,:] + V[Ind1,:]
                U2 = U[Ind2,:]
                           
                H = (U1.T * U1 + U2.T * U2) / 4 + Sigma1(i,i)*I_q;
       
                invH = H\I_q;
                invH = (invH+invH.T)/2;
                U(i,:) = U(i,:) + Gradient * invH; 
           
        for i in range(nSmp):
            
            S = zeros(nSmp,1);        
            Ind = find(Z(:,i)==1);
            X = U(Ind,:) * (U(i,:)+V(i,:)).T + muu;
            S(Ind) = exp(X/2)./(exp(-X/2)+exp(X/2));  
            
            Gradient = (A(:,i).T - S.T)*U - Sigma2(i,:)*V;
            
            U1 = U(Ind,:)
            G = (U1.T*U1)/4 + Sigma2(i,i) * I_q;
            
            invG = G\I_q
            invG = (invG+invG.T)/2;
            V(i,:) = V(i,:) + Gradient * invG;
        
        X = U * (U+V).T + muu;
        S = exp(X/2)./(exp(-X/2)+exp(X/2));    
        muu = muu+4/(sum(sum(Z))+4*Ta) * (sum(sum(A-Z.*S))-Ta*muu);
    



