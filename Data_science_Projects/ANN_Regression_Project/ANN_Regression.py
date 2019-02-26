#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dependencies

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:



def sigmoid(H):
    return 1/(1 + np.exp(-H))


def ReLU(h):
    return h*(h>0)


def tanh(h):
    return np.tanh(h)


def D_ReLU(z):
    return (z > 0)


def D_sigmoid(z):
    return z*(1-z)


def D_tanh(z):
    return 1-z*z


def creatWbs(self):
    
    self.Ws.append(np.random.randn(self.X.shape[1],self.N[0]))
    self.bs.append(np.random.randn(self.N[0]))
        
    if (len(self.N) > 1):
        for i in range(len(self.N)-1):
            self.Ws.append(np.random.randn(self.N[i],self.N[i+1]))
            self.bs.append(np.random.randn(self.N[i+1]))
            
    self.Ws.append(np.random.randn(self.N[-1],self.Y.shape[1]))
    self.bs.append(np.random.randn(self.Y.shape[1]))


def creat_DFs(R):
    Df = []
    for x in R.Fs:
        if (x == ReLU):
            Df.append(D_ReLU)
        if (x == sigmoid):
            Df.append(D_sigmoid)
        if (x == tanh):
            Df.append(D_tanh)
    R.DF = Df


def feed_forward(self,X):
        
    ZP = []
        
    ZP.append(self.Fs[0](np.matmul(X,self.Ws[0])+ self.bs[0])) 
        
    if (len(self.Fs) > 1):
        for i in range (1,len(self.Ws)-1):

            ZP.append(self.Fs[i](np.matmul(ZP[-1],self.Ws[i])+ self.bs[i]))
            
    ZP.append(np.matmul(ZP[-1],self.Ws[-1])+ self.bs[-1])
    
    return ZP


def OLS(y,y_hat):
    return np.sum((y-y_hat)**2)


        

class ANN_Regression():
    
    def __init__(self,x,y,N = [4], Fs = [ReLU], eta = 1e-2, epochs=100, mu = 0.9, gama = 0.999 ):
    
        if (len(N) == len(Fs)):
            
            if len(x.shape) == 1:
                x = x.reshape(x.shape[0],1)
                
            if len(y.shape) == 1:
                y = y.reshape(y.shape[0],1)
        
            self.X = x
            self.Y = y
            self.N = N
            self.Fs = Fs
            self.DF = [ReLU]
        
            self.Ws = []
            self.bs = []
            self.J = []
        #self.ZP = []
            self.P = []
            self.P_v = []  # perdiction P
        
            self.eta = eta
            self.epochs = epochs
            self.mu = mu
            self.gama = gama
            
        else :
            print("Layer and Functions are not the same size!")
    

    
    def Fit(self,lambda2 = 0, lambda1 = 0,epsilon = 1e-10, batch_sz = 100, show_curve = False):
        creatWbs(self)
    
        creat_DFs(self)
    
        if(len(self.DF) ==0):
            self.DF = [D_ReLU]
    
        t = 0
        
        mu = self.mu
        gama = self.gama
    
        GW = [1]*(len(self.N)+1)
        Gb =[1]*(len(self.N)+1)
        
        Mw = [0]*(len(self.N)+1)
        Mb = [0]*(len(self.N)+1)
    
        eta = self.eta
    
        for epoch in range (self.epochs):
            
            idx = np.random.permutation(self.X.shape[0])
            X = self.X[idx,:]
            Y = self.Y[idx,:]
        
            for i in range(self.X.shape[0]//batch_sz):
                x_i = X[i*batch_sz:(i+1)*batch_sz,:]
                y_i = Y[i*batch_sz:(i+1)*batch_sz,:]
            
                t += 1
            
                ZP_i = feed_forward(self,x_i)
            
                dH = []   #dh4,dh3,dh2....
                dW = []   #dW4,dW3,dW2....
                dZ = []   #dZ4,dZ3,dZ2....
                          #ZP = Z1,Z2,Z3...P
                          #Ws = W1,W2,W3...
                          #bs = b1,b2,b3....
            
                dH.append(ZP_i[-1] - y_i)
                dW.append(np.matmul(ZP_i[-2].T, dH[-1]))
                
                #db = dH[-1].sum(axis = 0) + lambda1*np.sign(self.bs[-1]) + lambda2*self.bs[-1]
                db = dH[-1].sum(axis = 0)

                dw = dW[-1] + lambda1*np.sign(self.Ws[-1]) + lambda2*self.Ws[-1]
                
                #Mw[-1] = mu*Mw[-1] + (1-mu)*dW[-1]
                Mw[-1] = mu*Mw[-1] + (1-mu)*dw
                Mb[-1] = mu*Mb[-1] + (1-mu)*db
        
                GW[-1] = gama*GW[-1] + (1-gama)*dw**2
                Gb[-1] = gama*Gb[-1] + (1-gama)*db**2
            
                self.Ws[-1] -= eta/np.sqrt(GW[-1]/(1-gama**t) + epsilon)*(Mw[-1]/(1-mu**t))
                self.bs[-1] -= eta/np.sqrt(Gb[-1]/(1-gama**t) + epsilon)*(Mb[-1]/(1-mu**t))
            
        
                dZ.append(np.matmul(dH[-1], self.Ws[-1].T))
            
                if (len(ZP_i)>2):
                    for i in range (len(ZP_i)-2,0,-1):
                        dH.append(dZ[-1]*self.DF[i](ZP_i[i]))
                        dW.append(np.matmul(ZP_i[i-1].T, dH[-1]))
                        
                        #db = dH[-1].sum(axis = 0)
                        #db = dH[-1].sum(axis = 0) + lambda1*np.sign(self.bs[i]) + lambda2*self.bs[i]
                        db = dH[-1].sum(axis = 0)
                        dw = dW[-1] + lambda1*np.sign(self.Ws[i]) + lambda2*self.Ws[i]
                    
                        #Mw[i] = mu*Mw[i] + (1-mu)*dW[-1]
                        Mw[i] = mu*Mw[i] + (1-mu)*dw
                        Mb[i] = mu*Mb[i] + (1-mu)*db
        
                        #GW[i] = gama*GW[i] + (1-gama)*dW[-1]**2
                        GW[i] = gama*GW[i] + (1-gama)*dw**2
                        Gb[i] = gama*Gb[i] + (1-gama)*db**2
            
                        self.Ws[i] -= eta/np.sqrt(GW[i]/(1-gama**t) + epsilon)*(Mw[i]/(1-mu**t))
                        self.bs[i] -= eta/np.sqrt(Gb[i]/(1-gama**t) + epsilon)*(Mb[i]/(1-mu**t))
                    
                        dZ.append(np.matmul(dH[-1], self.Ws[i].T))
            
                dH.append(dZ[-1]*self.DF[0](ZP_i[0]))
                dW.append(np.matmul(x_i.T, dH[-1]))
                
                #db = dH[-1].sum(axis = 0)
                #db = dH[-1].sum(axis = 0) + lambda1*np.sign(self.bs[0]) + lambda2*self.bs[0]
                db = dH[-1].sum(axis = 0)
                dw = dW[-1] + lambda1*np.sign(self.Ws[0]) + lambda2*self.Ws[0]
                
                #Mw[0] = mu*Mw[0] + (1-mu)*dW[-1]
                Mw[0] = mu*Mw[0] + (1-mu)*dw
                Mb[0] = mu*Mb[0] + (1-mu)*db
        
                #GW[0] = gama*GW[0] + (1-gama)*dW[-1]**2
                GW[0] = gama*GW[0] + (1-gama)*dw**2
                Gb[0] = gama*Gb[0] + (1-gama)*db**2
            
                self.Ws[0] -= eta/np.sqrt(GW[0]/(1-gama**t) + epsilon)*(Mw[0]/(1-mu**t))
                self.bs[0] -= eta/np.sqrt(Gb[0]/(1-gama**t) + epsilon)*(Mb[0]/(1-mu**t))
               
            
            if t % 100 == 0:
                P = feed_forward(self,self.X)[-1]
                self.J.append(OLS(self.Y,P) + lambda2/2 * sum(np.sum(W*W) for W in self.Ws) 
                              + lambda1 * sum(np.sum(np.abs(W)) for W in self.Ws))  #need to add L2 and L1
                #self.J.append(OLS(self.Y,P))
                
        if (show_curve):
            plt.plot(self.J)
        
        self.P = feed_forward(self,self.X)[-1]
    
    
    def R2 (self):
        return 1 - OLS(self.Y,self.P) / OLS(self.Y,self.P.mean())
 
    def R2_P (self,y):
        Y = y.reshape(y.shape[0],1)
        return 1 - OLS(Y,self.P_v) / OLS(Y,self.P_v.mean())
        
    def predict(self, x):
        if len(x.shape) == 1:
                x = x.reshape(x.shape[0],1)
        self.P_v = feed_forward(self,x)[-1]
        


# In[3]:


## Regeression With Adam


# In[ ]:




