import numpy as np
from scipy.linalg import lu  

def lu_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform LU decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            The permutation matrix P, lower triangular matrix L, and upper triangular matrix U.
    """
    P, L, U = lu(x) 
    return P, L, U

# Example Usage:
A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
P, L, U = lu_decomposition(A)

print("Permutation Matrix P:\n", P)
print("Lower Triangular Matrix L:\n", L)
print("Upper Triangular Matrix U:\n", U)

# We can confirm the decomposition is correct by multiplying the obtained matrices (A = U @ P @ L)
A_recover = np.round(P @ L @ U, 1) 
print("A_recover:\n", A)

"""
Program output:
Permutation Matrix P:
 [[0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]
Lower Triangular Matrix L:
 [[ 1.   0.   0. ]
 [ 0.5  1.   0. ]
 [-0.5  1.   1. ]]
Upper Triangular Matrix U:
 [[ 4. -6.  0.]
 [ 0.  4.  1.]
 [ 0.  0.  1.]]
PLU multiplication:
 [[ 2  1  1]
 [ 4 -6  0]
 [-2  7  2]]
 """

def qr_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform QR decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray]: The orthogonal matrix Q and upper triangular matrix R.
    """
    Q, R = np.linalg.qr(x)
    return Q, R

# Example usage
A = np.array([[1, 2], [3, 4], [5, 6]])
Q, R = qr_decomposition(A)

print("Matrix A:\n", A)
print("Orthogonal matrix Q:\n", Q)
print("Upper triangular matrix R:\n", R)
print("Verification (Q @ R = A):\n", Q @ R)

"""
Program output:
Matrix A:
 [[1 2]
 [3 4]
 [5 6]]
Orthogonal matrix Q:
 [[-0.16903085  0.89708523]
 [-0.50709255  0.27602622]
 [-0.84515425 -0.34503278]]
Upper triangular matrix R:
 [[-5.91607978 -7.43735744]
 [ 0.          0.82807867]]
Verification (Q @ R = A):
 [[1. 2.]
 [3. 4.]
 [5. 6.]]
"""

def determinant(x: np.ndarray) -> np.ndarray:
    """
    Calculate the determinant of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The determinant of the matrix.
    """
    return np.linalg.det(x)

# Example usage
A = np.array([[3, 1], [2, 5]])
det_A = determinant(A)
print("Determinant det(A):", det_A)

"""
Program output:
Determinant det(A): 13.0
"""

def eigen(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and right eigenvectors of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: The eigenvalues and the right eigenvectors of the matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eig(x)
    return eigenvalues, eigenvectors

# Example usage
A = np.array([[5,  3, 0],[2,  6, 0], [4,  -2, 2]])
w, v = eigen(A)

print("Matrix A:\n", A)
print("Eigenvalues (w):\n", w)
print("Eigenvectors (v):\n", v) 

# Verification : Let's take the first pair: eigenvalue w[0] and eigenvector v[:, 0]
lambda_1 = w[0]
v_1 = v[:, 0]

# The results of A * v and lambda * v must be the same
Av = A @ v_1 
Lambdav = lambda_1 * v_1
print(f"A * v: {Av}")
print(f"lambda * v: {Lambdav}")

"""
Program output:
Matrix A:
 [[ 5  3  0]
 [ 2  6  0]
 [ 4 -2  2]]
Eigenvalues (w):
 [2. 8. 3.]
Eigenvectors (v):
 [[ 0.          0.6882472   0.18291323]
 [ 0.          0.6882472  -0.12194215]
 [ 1.          0.22941573  0.97553722]]
A * v: [0. 0. 2.]
lambda * v: [0. 0. 2.]
"""


def svd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Singular Value Decomposition (SVD) of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The matrices U, S, and V.
    """
       U, S, Vt = np.linalg.svd(x, full_matrices=False)
    
    # Transpose Vt to get V
    V = Vt.T
    return U, S, V

# Example Usage:
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, V = svd(A)

print("U =\n", U)
print("S =\n", S)
print("V =\n", V)

# Reconstruct A
A_reconstructed = U @ np.diag(S) @ V.T
print("Reconstructed A =\n", A_reconstructed)

"""
Program output:
U =
 [[-0.2298477   0.88346102]
 [-0.52474482  0.24078249]
 [-0.81964194 -0.40189603]]
S =
 [9.52551809 0.51430058]
V =
 [[-0.61962948 -0.78489445]
 [-0.78489445  0.61962948]]
Reconstructed A =
 [[1. 2.]
 [3. 4.]
 [5. 6.]]
"""
