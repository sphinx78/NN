import numpy as np
# inputs =[1,2,3,2.5]
# weights = [[0.2,0.8,-0.5,1.0],
#               [0.5,-0.91,0.26,-0.5],
#                 [-0.26,-0.27,0.17,0.87]]
# bias=[2,3,0.5]

# outputs = []
# for n_weights, n_bias in zip(weights,bias):
#     output=0
#     for n_inputs,n_weights in zip(inputs, n_weights):
#         output += n_inputs*n_weights
#     output+=n_bias
#     outputs.append(output)

# print(outputs)


# output = np.dot(weights, inputs) + bias
# print(output)

# inputs = [[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8] ]
# weights = [[0.2, 0.8, -0.5, 1.0],[ 10.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87] ]
# biases = [2.0, 3.0, 0.5]

# weights2 = [[ 0.1 ,- 0.14 , 0.5 ],
# [ - 0.5 , 0.12 ,- 0.33 ],
# [ - 0.44 , 0.73 ,- 0.13 ]]
# biases2 = [ - 1 , 2 ,- 0.5 ]
# output=np.dot(inputs,np.array(weights).T) +biases
# print(output)
# all
# output_final=np.dot(output,weights2) + biases2
# print(output_final)
def create_data(points, classes):
    X = np.zeros( (points*classes, 2))
    y = np.zeros (points*classes, dtype='uint8') 
    for class_number in range(classes) : 
        ix = np.arange(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        # radius
        t = np. linspace(class_number*4, (class_number+1) *4, points) + np.random.randn(points)*0.2 
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)] 
        y[ix] = class_number 
    return X, y

# X = [[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8] ]

class Activation_ReLU:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)
class Activation_softmax:
    def forward(self,inputs):
        self.output=(np.exp(inputs-np.max(inputs,axis=1,keepdims=True)))/np.sum((np.exp(inputs-np.max(inputs,axis=1,keepdims=True))),axis=1,keepdims=True)
class Layer_dense:
    def __init__(self, input_n, neurons_n):
        self.weights= 0.1*np.random.rand(input_n,neurons_n)
        self.biases=np.zeros((1, neurons_n))
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.biases


X,y=create_data(100,3)
layer1=Layer_dense(2,3)
layer1.forward(X)
activation1=Activation_ReLU()
activation1.forward(layer1.output)


layer2=Layer_dense(3,3)
layer2.forward(activation1.output)
activation2=Activation_softmax()
activation2.forward(layer2.output)
output=activation2.output
print(output[:5])