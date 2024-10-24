import numpy as np

import numpy as np
def matrix(a,b):
    res = np.zeros((a.shape[0], b.shape[1]))
    for i in range(b.shape[1]):
        for j in range(a.shape[0]):
            for k in range(a.shape[1]):
                res[j,i] += a[j,k]*b[k,i]
    return res


