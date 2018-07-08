import numpy as np
import matplotlib.pyplot as pt

iter = 50000
learning_rate = 0.2

data = np.array([[3,1.5,1],
                [2,1,0],
                [4,1.5,1],
                [3,1,0],
                [3.5,0.5,1],
                [2,0.5,0],
                [5.5,1,1],
                [1,1,0]],dtype=np.float64)

#Separate inputs from outputs
x = data[:,[0,1]] 
y = data[:,2]

unknown = np.array([4.5, 1],dtype=np.float64)

w1 = np.random.random()
w2 = np.random.random()
b = np.random.random()

def sigmoid(x):
    return 1/(1+1+np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) - (sigmoid(x) **2)

#plot scatter
for i in range(0,len(x),1):
    color = ''
    if y[i] == 0:
        color = 'b'
    if y[i] == 1:
        color = 'r'                      
    pt.scatter(x[i,0],x[i,1],c=color)
pt.grid()
pt.axis([0,6,0,2])

        
#Training loop
#Iterate over timestep
derivSSE_w1 = 0
derivSSE_w2 = 0
derivSSE_b = 0
N = float(len(data))

for i in range(0,iter,1):

    #iterate over data series
    SSE = 0
    for j in range (0,len(data),1):
        z = x[j,0]*w1 +x[j,1]*w2 + b
        y_mod = sigmoid(z)
        SSE += (y[j] - y_mod)**2
        #print('At point x(0)=',x[j,0],'SSE=',SSE)
        derivSSE_w1 += (-2/N) * (y[j]-y_mod) * x[j,0]
        derivSSE_w2 += (-2/N) * (y[j]-y_mod) * x[j,1]
        derivSSE_b += (-2/N) * (y[j]-y_mod)
    w1 = w1 - (derivSSE_w1*learning_rate)
    w2 = w2 - (derivSSE_w2*learning_rate)
    b = b - (derivSSE_b*learning_rate)
    
        
        
#        if i%100 == 0:
#            print('At point x(0)=',x[j,0],'SSE=',SSE)


    if i%1000 ==0:
        print('MSE at i of ',i,'is =',SSE/N)
        print('w1=',w1,'w2=',w2,'b=',b)




    

