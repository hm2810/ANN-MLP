import numpy as np

#hyperparameters
hid_layer_neurons = 3  #No. of neurons in hidden layer
learningRate = 0.01
epoch = 1000

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)  #Row-> Samp, Colums = Para
y = np.array(([75], [82], [93]), dtype=float)      #Row-> Samp, Colums = Para

#Normalise 
X = X/np.amax(X, axis=0) # axis=0 will ensure each para in matrix is normalised individually
y = y/100 #Max y is 100 (not in training data)


class Neural_Network(object):
    
    #Create architecture of network
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = len(X[0]) #no. of input neurons are columns of x
        self.hiddenLayerSize = hid_layer_neurons  #hyper para
        self.outputLayerSize = len(y[0]) #no. of output neurons are columns of y 
                
        #Weights (parameters)   W1-> Layer 1, W2 -> Layer 2
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)  
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
     
    #Define operations in forward operation
    def forward(self, X):
        #Propagate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
    
    #Define activation and derivative functions
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z)) 
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
        
    #Define cost function. Wont be used for training and it is summed and averaged.
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        N = len(y[0])
        J = (1/N)*sum((y-self.yHat)**2)
        return J
    
    #Define cost function primes ---->> BACK PROPAGATION
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-2*(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2

##------------------------------------------------------------------------------    
##Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into single column vector:
        global params
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        print('Ran getParams')
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        print('Ran setParams')

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
        print('Ran computerGradients')
#    
#def computeNumericalGradient(N, X, y):
#    paramsInitial = N.getParams()
#    numgrad = np.zeros(paramsInitial.shape)
#    perturb = np.zeros(paramsInitial.shape)
#    e = 1e-4
#
#    for p in range(len(paramsInitial)):
#        #Set perturbation vector
#        perturb[p] = e
#        N.setParams(paramsInitial + perturb)
#        loss2 = N.costFunction(X, y)
#        
#        N.setParams(paramsInitial - perturb)
#        loss1 = N.costFunction(X, y)
#
#        #Compute Numerical Gradient
#        numgrad[p] = (loss2 - loss1) / (2*e)
#
#        #Return the value we changed to zero:
#        perturb[p] = 0
#        
#    #Return Params to original value:
#    N.setParams(paramsInitial)
#
#    return numgrad
#  
##------------------------------------------------------------------------------

#Run NN
NN = Neural_Network()

for i in range(epoch):
    cost1 = NN.costFunction(X,y)
    dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
    
    NN.W1 = NN.W1 - dJdW1 * learningRate
    NN.W2 = NN.W2 - dJdW2 * learningRate
    cost2 = NN.costFunction(X,y)
    
    if i%(epoch/10) == 0:
        print ('DerivW1=',sum(dJdW1/len(dJdW1)),'DerivW2=',sum(dJdW2/len(dJdW2)),'Cost=',cost2)
 
    
##------------------------------------------------------------------------------   
## Run diagnostics for gradients
#numgrad = computeNumericalGradient(NN, X, y)
#grad = NN.computeGradients(X,y)
#
#normgrad = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)
#print('normgrad=',normgrad)
##------------------------------------------------------------------------------        
        
#Run Trainer       
from scipy import optimize

class trainer(object):
    def __init__(self, N):
        #Make Local reference to neural network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        print('Ran callbackF')
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        print('Ran costFunctionWrapper')
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)
        self.N.setParams(_res.x)
        global OptimizationResults
        OptimizationResults = _res
        print('Ran train')
 
       
T = trainer(NN) 
T.train(X,y)

import matplotlib.pyplot as plt
plt.plot(T.J)
plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Cost')




#------------------------------------------------------------------------------
#Test network for various combinations of sleep/study:
hoursSleep = np.linspace(0, 10, 100)
hoursStudy = np.linspace(0, 5, 100)

#Normalize data (same way training data way normalized)
hoursSleepNorm = hoursSleep/10.
hoursStudyNorm = hoursStudy/5.

#Create 2-d versions of input for plotting
a, b  = np.meshgrid(hoursSleepNorm, hoursStudyNorm)

#Join into a single input matrix:
allInputs = np.zeros((a.size, 2))
allInputs[:, 0] = a.ravel()
allInputs[:, 1] = b.ravel()

allOutputs = NN.forward(allInputs)

#Contour Plot:
yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T

CS = plt.contour(xx,yy,100*allOutputs.reshape(100, 100))
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('Hours Sleep')
plt.ylabel('Hours Study')

#3D plot:

##Uncomment to plot out-of-notebook (you'll be able to rotate)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xx, yy, 100*allOutputs.reshape(100, 100),cmap=matplotlib.cm.jet)

ax.set_xlabel('Hours Sleep')
ax.set_ylabel('Hours Study')
ax.set_zlabel('Test Score')