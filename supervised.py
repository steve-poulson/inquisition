from sklearn import ensemble
from numpy import *
from numpy.linalg import *

class supervised():
    def __init__(self, A):
        self.A = A
        self.A2 = A*A
        self.A3 = A*A*A
        
        self.N = len(A)
        self.X = []
        self.Y = []
        
        
    def features(self,i,j):
        return [[(self.A[i,:] * self.A[:,j])[0,0] / (norm(self.A[i,:]) * norm(self.A[j,:])),self.A3[1,j],self.A2[i,j]]]
    
    def train(self,i,j,y):
        
        self.X  +=  self.features(i,j)
        self.Y += [y]
        
        clf = ensemble.RandomForestClassifier()
        
        print 'X',self.X
        print 'Y',self.Y
        
        clf = clf.fit(self.X, self.Y)
        
        Y1 = zeros((self.N,self.N))
        
        for i in range(self.N):
            for j in range(self.N): 
                Y1[i,j] = clf.predict(self.features(i,j))
        
        return Y1 