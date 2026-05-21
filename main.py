import numpy as np


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

class Loss:
    def calculate(self, output,y):
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss
    
class CategoricalCrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        y_pred_clipped=np.clip(y_pred,1e-7,1-1e-7)

        if len(y_true.shape)==1:
            correct_confidence = y_pred_clipped[range(samples),y_true]
        if len(y_true.shape)==2:
            correct_confidence=np.sum(y_true*y_pred_clipped,axis=1)

        neg_log_likelihood=-np.log(correct_confidence)
        return neg_log_likelihood
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Activation_SoftmaxLoss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = CategoricalCrossEntropy()
    

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
  
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        self.output = (np.exp(inputs - np.max(inputs, axis=1, keepdims=True))) / np.sum((np.exp(inputs - np.max(inputs, axis=1, keepdims=True))), axis=1, keepdims=True)
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
class Layer_Dense:

    def __init__(self, input_n, neurons_n):
        self.weights = 0.1 * np.random.randn(input_n, neurons_n)
        self.biases = np.zeros((1, neurons_n))
    
    def forward(self, inputs):
        self.inputs=inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Optimizer_SGD:
    def __init__(self,learning_rate=1,decay=0,momentum=0):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.momentum=momentum
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_params(self,layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums=np.zeros_like(layer.weights)
                layer.biases_momentums=np.zeros_like(layer.biases)
            weight_updates=self.momentum*layer.weight_momentums-self.current_learning_rate*layer.dweights
            biases_updates=self.momentum*layer.biases_momentums-self.current_learning_rate*layer.dbiases
            layer.weight_momentums = weight_updates      # missing
            layer.biases_momentums  = biases_updates     # missing
        else:
            weight_updates=-self.current_learning_rate * layer.dweights
            biases_updates=-self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += biases_updates

    def post_update_params(self):
        self.iterations+=1

class Optimizer_Adagrad:
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

X, y = create_data(100, 3)

dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_SoftmaxLoss_CategoricalCrossentropy()
optimizer=Optimizer_Adagrad(decay = 1e-4)


for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)


    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f"epoch: {epoch},"+f"accuracy={accuracy: 3f}"+f"loss={loss: 3f}"+f"learning rate={optimizer.current_learning_rate}")

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
