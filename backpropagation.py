import numpy as np

x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0 

output= np.dot(x,w)+b
print(output)

#activation Relu
y=max(output,0)
print(y)