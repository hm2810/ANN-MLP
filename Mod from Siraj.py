import numpy as np

#Hyperparameters
iter = 10000
neurons = 5

#sigmoid
def sigmoid(x):
     return 1/(1+np.exp(-x))

#derivative
def derivative(x):
    return (x*(1-x))

#input for training
x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1],
              [0.5,0.8,0.4]],dtype=np.float)

y = np.array([[0],
              [1],
              [1],
              [0],
              [0.5]],dtype=np.float)

#seed
np.random.seed(1)

#weights and bias
b = 1
w0_1 = 2*np.random.random((len(x[0]),neurons)) - b #Connecting layer 0 to layer 1
w1_2 = 2*np.random.random((neurons,len(y[0]))) - b #Connecting later 1 to layer 2

print('Initial weight w0:')
print(w0_1)
print('Initial weight w1:')
print(w1_2)

#training
for i in range(0,iter,1):
    #layers
    l0 = x
    l1 = sigmoid(np.dot(l0,w0_1))
    l2 = sigmoid(np.dot(l1,w1_2))

    #backpropagation
    l2_error = y -l2
    
    #print error
    if (i%1000) == 0:
        print('Error after '+ str(i) + ' is:' + str(np.mean(np.abs(l2_error))))
    
    #calculate deltas
    l2_delta = l2_error * derivative(l2)
    l1_error = np.dot(l2_delta, w1_2.T)
    l1_delta = l1_error * derivative(l1)
    
    #update weights
    w1_2 += np.dot(l1.T,l2_delta)
    w0_1 += np.dot(l0.T, l1_delta)
    
print('Predicted result is')    
print (l2)
print('Final weight w0:')
print(w0_1)
print('Final weight w1:')
print(w1_2)