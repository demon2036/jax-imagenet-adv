import numpy as np


x=np.array([[1,2,3,4,5]],dtype=np.float32)
y=np.array([[0,1,0,1,1]]).T

print(x.shape,y.shape)
print(x)
print(y)

x[y.T==0]=np.nan

print(x)