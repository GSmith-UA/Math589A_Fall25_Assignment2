import numpy as np

def paqlu_decomposition_in_place(A):
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
        if np.abs(U[k,k]) < 1e-12:
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


def solve(A, b):
    P,L,U,Q = paqlu_decomposition_in_place(A)
    m,n = np.shape(A)
    # The rank can be pulled from either the total columns of L or the total rows of U
    r = np.shape(U)[0]


    # Probably need to permute b so I'm going to hit it with P
    b_perm = P@b

    y = forwardSubstitution(L,b_perm[:r])
    if np.any(np.abs(b_perm[r:])) > 1e12:
        raise ValueError("System is inconsistent")
    x_part = backSubstitution(U,y)

    x_part = Q@x_part # Re order the columns

    #We need to construct general solution to Ux = 0
    N = []
    # The remaining columns in U are the free variables
    for i in range(r,n):
        x_null = np.zeros((n,1))
        x_null[i] = 1 # Set variable of interest to 1 and other free var to 0

        for k in range(r-1,-1,-1):
            x_null[k] = -np.dot(U[k,k+1:],x_null[k+1:]) / U[k,k]
        N.append(Q@x_null)
        
    # if N is non-empty stack into a matrix
    if N:
        N = np.hstack(N)
    else:
        N = np.zeros((n,0))

    return x_part,N

def backSubstitution(U,y):
    # U should be Upper Triangular
    rank,colNum = np.shape(U)
    x = np.zeros((colNum,1),dtype = float) # Pre-allocate solution vector

    for i in range(rank - 1,-1,-1):
        if np.abs(U[i,i])<1e-12:
            raise ValueError("Singular Matrix in back sub")
        x[i] = (y[i] - (np.dot(x[i+1:].flatten(),U[i,i+1:])))/U[i,i]
    return x

def forwardSubstitution(L,b):
    # L should be Lower Triangular
    rank = np.shape(L)[1]
    x = np.zeros((rank,1),dtype = float) # Pre-allocate solution vector

    for i in range(0, rank):
      if np.abs(L[i,i])<1e-12:
        raise ValueError("Singular Matrix in forward sub")
      x[i] = (b[i] - (np.dot(x[:i].flatten(),L[i,:i])))/L[i,i]
    return x

