import numpy as np

def paqlu_decomposition_in_place(A,TOL=1e-12):
    m = A.shape[0] # Grab the number of rows...
    n = A.shape[1] # Grabs the number of columns

    # The rank must be less than or equal to the minimum of the rows and columns
    max_rank = min(m,n)
    rank = max_rank # If we end up rank deficient I will overwrite this

    P = np.eye(m)
    Q = np.eye(n)

    # I'm pre-allocating with the maximum the rank can be...
    # If we are rank deficient then I can clip things later...
    L = np.zeros((m,max_rank),dtype = float)
    U = A.astype(float).copy()

    for k in range(0,max_rank):

        idx = np.argmax(np.abs(U[k:,k:]))
        pivot_row = k + (idx//U[k:,k:].shape[1])
        pivot_col = k + (idx%U[k:,k:].shape[1])

        # Swap rows k and r in U
        U[[pivot_row,k],:] = U[[k,pivot_row],:]
        # Swap rows k and r in P
        P[[pivot_row,k],:] = P[[k,pivot_row],:]

        U[:,[pivot_col,k]] = U[:,[k,pivot_col]]
        Q[:,[pivot_col,k]] = Q[:,[k,pivot_col]]
        # Swap rows k and r in L (but only columns 1:k-1) or in python 0:k-1
        if k-1>=0:
            L[[pivot_row,k],0:k] = L[[k,pivot_row],0:k]

        # If the diagonal element is now close to zero... call us rank deficient and exit
        if np.abs(U[k,k]) < TOL:
            rank = k
            break

        for i in range(k+1,m):
            L[i,k] = U[i,k]/U[k,k]
            U[i,k:] = U[i,k:] - L[i,k]*U[k,k:]
            

    for i in range(0,rank):
        L[i,i] = 1

    L = L[:,:rank]
    U = U[:rank,:]
    # Note: P must be a vector, not array
    # return P, Q, A
    return P, L, U, Q


