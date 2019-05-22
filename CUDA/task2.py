import numpy as np
from numba import cuda

TPB = 16
@cuda.jit
def matmul(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    if x >= C.shape[0] and y >= C.shape[1]:
        return
    tmp = 0.
    for i in range(bpg):
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]
        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        cuda.syncthreads()
    C[x, y] = tmp


N = 500
A = np.random.rand(N,N)
B = np.random.rand(N,N)


import timeit

np_loop = 300
np_matmul = np.empty(np_loop)


for i in range(np_loop):
    t_start = timeit.default_timer()
    out = np.matmul(A, B)
    t_end = timeit.default_timer()
    np_matmul[i] = t_end - t_start

record = np_matmul
print ("np.matmul(A, B) takes average {:.5f} second (except 1st run)".format(np_matmul[1:].mean()))
print ("{:<10}{:<10}{:<10}{:<10}".format("mean","max","min","std"))
print ("{:<10.5f}{:<10.5f}{:<10.5f}{:<10.5f}".format(record.mean(),record.max(),record.min(),record.std()))
print ("record")
print (record)