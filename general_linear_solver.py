import numpy as np

def paqlu_decomposition_in_place(A,TOL=1e-6):
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

        idx = np.argmax(np.abs(U[k:,k:])) # This grabs the linear index in the submatrix. I unravel it below... 
        pivot_row = k + (idx//U[k:,k:].shape[1])
        pivot_col = k + (idx%U[k:,k:].shape[1])

        # Here we check the magnitude of the pivot before we perform the switches...
        if np.abs(U[pivot_row,pivot_col]) < TOL:
            rank = k
            break

        # Swap rows k and r in U
        U[[pivot_row,k],:] = U[[k,pivot_row],:]
        # Swap rows k and r in P
        P[[pivot_row,k],:] = P[[k,pivot_row],:]

        # Swap columns
        U[:,[pivot_col,k]] = U[:,[k,pivot_col]]
        Q[:,[pivot_col,k]] = Q[:,[k,pivot_col]]

        # Swap rows k and r in L (but only columns 1:k-1) or in python 0:k-1
        if k-1>=0:
            L[[pivot_row,k],0:k] = L[[k,pivot_row],0:k]

        for i in range(k+1,m):
            L[i,k] = U[i,k]/U[k,k]
            U[i,k:] = U[i,k:] - L[i,k]*U[k,k:]
            
    # Might be faster if we just grabbed the slice and set to a one vector?
    # Can experiment with this later
    for i in range(0,rank):
        L[i,i] = 1

    # Here we clip L and U so that the dimensions are limited by the rank
    L = L[:,:rank]
    U = U[:rank,:]

    # At somepoint run some testing with a matrix P,Q vs a vector P,Q
    return P, L, U, Q


def solve(A, b, TOL=1e-6):
    # Solves a problem Ax = b
    # Returns a Null space if some variables are free along with a particular solution

    m,n = np.shape(A)

    # Edge cases
    if np.shape(b)[0] != np.shape(A)[0]:
        raise ValueError("b must have the same number of rows as A")
    
    if (m == 0):
        c = np.zeros((n,1),dtype=float)
        N = np.eye(n)
        return c,N
    if (n ==0):
        raise ValueError("There are no columns in the matrix A")

    # Perform the PAQ = LU decomp
    P,L,U,Q = paqlu_decomposition_in_place(A)

    # The rank can be pulled from either the total columns of L or the total rows of U
    r = np.shape(U)[0]

    # Construct the Null space (even if we pass multiple b vectors we only need to do this once)
    N = constructNullSpace_FromLU(U)

    # This is just in case we pass multiple b vectors (Marek doesn't do this...)
    b_num = np.shape(b)[1]

    # Initialize an empty array for the particular solutions
    c = []

    for i in range(0,b_num):
        # Performs Ax = b solve on all the b vectors given....

        # Probably need to permute b so I'm going to hit it with P
        b_perm = P@b[:,i]

        y = forwardSubstitution(L,b_perm[:r])

        # Check for inconsistency: all values in the y vector should be zero underneath the rank!
        if r < m:
            if np.any(np.abs(y[r:]) > TOL):
                return None, Q@N # Change this back to Value Error when you are done passing Marek's unit tests
                #raise ValueError("inconsistent system: A x = b has no solution")
        
        c1= backSubstitution(U,y)
        c.append(c1)

    if c:
        c = np.hstack(c)

    # I apply Q right before I return... got to save on the matrix mults... only do them when we have to   
    return Q@c,Q@N

def backSubstitution(U,y,TOL=1e-6):
    # U should be Upper Triangular
    rank,colNum = np.shape(U)
    x = np.zeros((colNum,1),dtype = float) # Pre-allocate solution vector

    for i in range(rank - 1,-1,-1):
        if np.any(np.abs(U[i,i])<TOL):
            raise ValueError("Singular Matrix in back sub")
        x[i] = (y[i] - (np.dot(x[i+1:].flatten(),U[i,i+1:])))/U[i,i]
    return x

def forwardSubstitution(L,b,TOL=1e-6):
    # L should be Lower Triangular
    rank = np.shape(L)[1]
    x = np.zeros((rank,1),dtype = float) # Pre-allocate solution vector

    for i in range(0, rank):
      if np.any(np.abs(L[i,i])<TOL):
        raise ValueError("Singular Matrix in forward sub")
      x[i] = (b[i] - (np.dot(x[:i].flatten(),L[i,:i])))/L[i,i]
    return x

def constructNullSpace_FromLU(U,TOL=1e-6):
    #We need to construct general solution to Ux = 0
    N = []
    r,n = np.shape(U)
    # The remaining columns in U are the free variables
    for i in range(r,n):
        x_null = np.zeros((n,1))
        x_null[i] = 1 # Set variable of interest to 1 and other free var to 0

        for k in range(r-1,-1,-1):
            # if np.any(np.abs(U[k,k]) < TOL):
            #     raise ZeroDivisionError("Calculating Nullspace: U has a pivot near zero")
            x_null[k] = -np.dot(U[k,k+1:],x_null[k+1:]) / U[k,k]
        N.append(x_null)
        
    # if N is non-empty stack into a matrix
    if N:
        N = np.hstack(N)
    else:
        N = np.zeros((n,0))
    
    return N

