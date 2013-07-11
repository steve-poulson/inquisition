"""
A log-linear model with latent features for dyadic prediction

A stochastic gradient implementation of the LFL model may be found here. The code assumes that there are structures Tr and Te for the train and test set, each comprising three vectors i,j, and r for the "user", "movie", and "rating" respectively. (As noted in the paper, these may be replaced with more general dyadic entities.) The code handles both nominal and ordinal "ratings". Sample usage:

k = 10; % # of latent features
eta0 = 0.01; % learning rate
lambda = 1e-6; % regularization parameter
epochs = 10; % # of sweeps over training set
loss = 'mse'; % loss function on training set

[w, trainErrors, testErrors] = lflSGDOptimizer(Tr, Te, k, eta0, lambda, epochs, loss);
"""

from numpy import *
from numpy.random import *
from numpy.linalg import *

def logLoss(p,r,R):
    p_ = [1 if a == r else 0 for a in range(0, R)]
    G = p - p_
    
    print "p",p
    print "G",G,
    print "p",R
    
    return G

def mseLoss(p,p_,R):
    
    #Take expected rating as the prediction
    prediction = sum(range(0,R) * p)       
    s = 2 * (prediction - r);         
    G = array(s * p * (range(0,R) - prediction))
    
    print G
    
    return G
     

class LFL:
    
    def __init__(self,A):
        self.xs = []
        self.ys = []
        self.ratings = []
        
        self.k=3
        self.eta=0.01
        self.lmbda=0
        self.epochs=15

        self.N = len(A)
        
        self.xW = None
        self.yW = None
    
    def train(self,i,j,y):
        
        loss = logLoss
        
        Y = zeros((self.N,self.N))
        
        self.xs += [i]
        self.ys += [j]
        
        self.xs += [j]
        self.ys += [i]
        
        
        self.ratings += [y,y]
        
        print self.xs,self.ys
    
         # todo:randomise
        n = len(self.xs);
    
        U = max(self.xs)  +1;
        M = max(self.ys)  + 1;
        R = max(self.ratings)  + 1;
    
        self.xW  = 1/self.k * standard_normal((self.k, R, U))
        self.yW  = 1/self.k * standard_normal((self.k, R, M))
        
        self.xW [0,:,:] = 1;
        self.yW [-1,:,:] = 1;
        #self.yW[:,1,:] = 0;
        
        
        #userW[0:-1,1:,:] = tile(userW[0:-1,0,:], (1,R-1,1))
        
        for e in range(1,self.epochs):
            
            print "epoch %d"%e
            
            etaCurr = self.eta/e
    
            for index in range(0,n):
                
                #defines edge
                x = self.xs[index]
                y = self.ys[index]
                r = self.ratings[index]
                
                
                xW = self.xW[:,:,x]
                yW = self.yW[:,:,y]
                
                xW[0,:] = 1
                yW[-1,:] = 1;    
    
                #% Vector whose ith element is Pr[rating = i | x, y; w]
                 
                p = exp(diag(dot(xW.T , yW)));
                p = p/sum(p);
                p = mat(p).H
                 
                G = loss(p,r,R)
                # seems to be error here instead of (1,n) get (n,n)
                Gx = yW * G
                Gy = xW * G
                
                # Regularization
                Gx[1:-1,:] = Gx[1:-1,:] + self.lmbda * xW[1:-1,:]
                Gy[1:-1,:] = Gy[1:-1,:] + self.lmbda * yW[1:-1,:]
                
                Gx[0,:] = 0;
                Gy[-1,:] = 0;
    
                Gx[:,0] = 0;
                Gy[:,0] = 0;
                
                Gx[0:-1,:] = tile(sum(Gx[0:-1,:], 1), (1,R))
    
                self.xW[:,:,x] = xW - etaCurr * Gx;
                self.yW[:,:,y] = yW - etaCurr * Gy;              
        
        self.xW[0,:,:] = 1;
        self.yW[-1,:,:] = 1;
        
        a,b,c = self.predict()
        
        return a
    
    """
% Predictions from the LFL model using the given weight vector
% This is computed over all users and movies (which are implicit in the weight)
% Output consists of the real-valued predictions (expected value under the
% probability model); discrete valued  'argmax predictions', viz the most
% likely label under the probability model; and the actual probabilities
% themselves
    """
    
    def predict(self):
    
        R = size(self.xW, 1);
        U = size(self.xW, 2);
        M = size(self.yW, 2);
        
        probabilities = zeros((U, M, R));
        
        for r in range(0,R):
            xW = squeeze(self.xW[:,r,:])
            yW = squeeze(self.yW[:,r,:])
            
            probabilities[:,:,r] = exp(dot(xW.T , yW))
            
            print self.xW.shape, xW.shape, probabilities.shape
        
        sumP = probabilities.sum(2)
        
        for r in range(0,R):
            probabilities[:,:,r] = probabilities[:,:,r] / sumP
        
        a = reshape(range(0,R), (1,1,R)) * probabilities
    
        predictions = sum(a, 2);
        
        argmaxPredictions = probabilities.argmax(2)
        return [predictions, argmaxPredictions, probabilities]

