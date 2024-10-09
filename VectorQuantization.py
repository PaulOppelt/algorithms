import numpy as np
import matplotlib.pyplot as plt

def VG(train_x, n_v,eps, epoches=5):
    train_x = train_x.reshape(train_x.shape[0],-1)
    init_vectors = np.random.randn(n_v,train_x.shape[1])
    for _ in range(epoches):
        for i,j in enumerate(train_x):
            vec =  np.sqrt(np.sum((init_vectors-j.reshape(1,-1))**2,axis=1))
            idx = np.argmin(vec)
            init_vectors[idx] = init_vectors[idx]- eps*(init_vectors[idx]-j.reshape(1,-1))
    return init_vectors

# create data and visualize
test_set = np.random.randn(10000,2)
plt.plot(test_set[:,0], test_set[:,1],".",color="green")
v = VG(test_set,30,0.01, epoches=20)
plt.plot(v[:,0], v[:,1],".",color="red")
