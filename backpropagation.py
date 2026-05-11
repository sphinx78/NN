import numpy as np

x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0 




#forward pass 
xw0 = x[0]*w[0]
xw1 = x[1]*w[1]
xw2 = x[2]*w[2]

#adding weighted input and bias
z = xw0 +xw1 +xw2 + b


#activation Relu
y=max(z,0)

#derivative from next layer
d_value=1

drelu_dz= d_value * (1. if z>0 else 0.)
print(drelu_dz)
dsum_dxw0=1
dsum_dxw1=1
dsum_dxw2=1
dsum_db=1

drelu_dxw0=drelu_dz*dsum_dxw0
drelu_dxw1=drelu_dz*dsum_dxw1
drelu_dxw2=drelu_dz*dsum_dxw2
drelu_db=drelu_dz*dsum_db

print(drelu_dz,drelu_db,drelu_dxw0,drelu_dxw1,drelu_dxw2)

# The partial derivative of f with respect to x equals y . The partial derivative of f with respect to y
# equals x . Following this rule, the partial derivative of the first weighted input with respect to the
# input equals the weight (the other input of this function). Then, we have to apply the chain rule
# and multiply this partial derivative with the partial derivative of the subsequent function, which is
# the sum (we just calculated its partial derivative earlier in this chapter):

dmul_dx0=w[0]
dmul_dx1=w[1]
dmul_dx2=w[2]
dmul_dw0=x[0]
dmul_dw1=x[1]
dmul_dw2=x[2]
dmul_db=1
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dx2 = drelu_dxw0 * dmul_dx2
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dw2 = drelu_dxw2 * dmul_dw2
drelu_db = drelu_db * dmul_db

print(drelu_dw0,drelu_dw1,drelu_dw2,drelu_dx0,drelu_dx1,drelu_dx2,dmul_dw2)



