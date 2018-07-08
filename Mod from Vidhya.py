#Import libraries
import numpy as np

#Hyper parameters
epoch = 10000
learning_rate = 0.1

#Generate data
x = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
y = np.array([[1],[0],[0]])

#Functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    return x*(1-x)

#Define neurons
Neu_in = x.shape[1] #number of neurons equal to no. of columns in x
Neu_hid = 3 #Neurons in hidden layer
Neu_out = y.shape[1] #number of neurons equal to no. of columns in y

#Weight and bias initialise
w_in = np.random.random(size=(Neu_in,Neu_hid))
b_in = np.random.random(size=(1,Neu_hid))
w_out = np.random.random(size=(Neu_hid,Neu_out))
b_out = np.random.random(size=(1,Neu_out))

for i in range(epoch):
    #Forward prop
    sig_hid = np.dot(x,w_in) + b_in #input to hidden layer
    act_hid = sigmoid(sig_hid) # activation of hidden layer
    sig_out = np.dot(act_hid,w_out) + b_out #output of hidden layer
    y_pred = sigmoid(sig_out) #Predicted output
    
    #Back prop
    Cost = y - y_pred
    
    deriv_out = deriv_sigmoid(y_pred) #Derivative of outer layer
    deriv_hid = deriv_sigmoid(act_hid)  #Derivative of hidden layer
    
    delta_out = Cost * deriv_out
    Cost_hid = np.dot(delta_out,sig_out.T)
    delta_hid = Cost_hid * deriv_hid
    
    w_out += np.dot(act_hid.T,delta_out) * learning_rate
    b_out += np.sum(delta_out, axis=0,keepdims=True) * learning_rate
    w_in +=   np.dot(x.T,delta_hid) * learning_rate
    b_in +=   np.sum(delta_hid, axis=0,keepdims=True) * learning_rate
    
    print(y_pred)



#for i in range(epoch):
#
##Forward Propogation
#hidden_layer_input1=np.dot(X,wh)
#hidden_layer_input=hidden_layer_input1 + bh
#hiddenlayer_activations = sigmoid(hidden_layer_input)
#output_layer_input1=np.dot(hiddenlayer_activations,wout)
#output_layer_input= output_layer_input1+ bout
#output = sigmoid(output_layer_input)
#
##Backpropagation
#E = y-output
#slope_output_layer = derivatives_sigmoid(output)
#slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
#d_output = E * slope_output_layer
#Error_at_hidden_layer = d_output.dot(wout.T)
#d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
#wout += hiddenlayer_activations.T.dot(d_output) *lr
#bout += np.sum(d_output, axis=0,keepdims=True) *lr
#wh += X.T.dot(d_hiddenlayer) *lr
#bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
#
#print output