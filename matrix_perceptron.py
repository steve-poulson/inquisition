from scipy.linalg import norm, eig, svd, pinv
from scipy.sparse.linalg import svds, eigs
from numpy import *
import matplotlib.pyplot as plt

class matrix_perceptron():
    
    def incidence(self,A):
        
        d = sum(A.getA(), 0)
        D = diag(d) 
        L = D - A 
        
        N = len(L)
        M = int(sum(d) / 2)
        
        
        print N,M
        Ai = zeros((M,N));
        
        l = 0;
        
        for i in range(N):
            for j in range(i+1,N):
                if L[i,j]!=0:
                    Ai[l,i] = 1
                    Ai[l,j] = -1
                    l = l + 1            
                    
        psi =  mat(Ai)
        
        return N,M,psi
    
    def calcX(self,i,j):
        e = mat(zeros(self.N)).H
        e[i] = 1
        e[j] = -1
        """
        plt.imshow(self.psi_inv.H * e,interpolation='nearest')
        plt.colorbar()
        plt.show();
        """
        return self.psi_inv.H * e * e.H * self.psi_inv
                
    def calcR(self):
        r = 0
        for i in range(self.N):
            for j in range(self.N):
                r =  max(r,trace(self.calcX(i,j)))
        return r
    
    def __init__(self,A):
        
        self.N,self.M,psi = self.incidence(A)
        
        
        self.psi_inv = mat(pinv(psi))
        
        R = self.calcR()
        self.theta =1/ (R*R)
        print "theta",self.theta
        self.W = mat(zeros((self.M,self.M))) 
        
        print self.psi_inv
        plt.imshow(self.psi_inv,interpolation='nearest')
        plt.colorbar()
        plt.show();
  
    def train(self,i,j,y):
       
        X = self.calcX(i,j)

        yp = 1 if trace(self.W.H * X) > self.theta else 0
        print "??",yp, y,trace(self.W.H * X)
        if yp != y:
            print "mistake"
            self.W += (y-yp)*X
                
        Y = zeros((self.N,self.N))
        
        for i in range(self.N):
            for j in range(self.N):
                #print trace(self.W.H * self.calcX(i,j)), self.theta
#                Y[i,j] =  1 if trace(self.W.H * self.calcX(i,j)) > self.theta else 0
                Y[i,j] =  trace(self.W.H * self.calcX(i,j))
                
        return Y
        
     
           