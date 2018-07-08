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
    
    #Define cost function primes
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-2*(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2

NN = Neural_Network()

for i in range(epoch):
    cost1 = NN.costFunction(X,y)
    dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
    
    NN.W1 = NN.W1 - dJdW1 * learningRate
    NN.W2 = NN.W2 - dJdW2 * learningRate
    cost2 = NN.costFunction(X,y)
    
    if i%(epoch/10) == 0:
        print ('Deriv1=',sum(dJdW1/len(dJdW1)),'Deriv2=',sum(dJdW2/len(dJdW2)),'Cost=',cost2)
    