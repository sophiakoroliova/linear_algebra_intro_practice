import numpy as np


def get_matrix(n: int, m: int) -> np.ndarray:
    """Create random matrix n * m.

    Args:
        n (int): number of rows.
        m (int): number of columns.

    Returns:
        np.ndarray: matrix n*m.
    """
    return np.random.rand(n, m)

#приклад використання функції 
#print(get_matrix(3, 4))


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrix addition.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: matrix sum.
    """
    return x + y

#приклад використання функції
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
#print(add(a, b))


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Matrix multiplication by scalar.

    Args:
        x (np.ndarray): matrix.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied matrix.
    """
    return x * a #перемножуємо кожен елемент матриці x на скаляр a

#приклад використання функції
x = np.array([[1, 2], [3, 4]])
a = 2.5
result = scalar_multiplication(matrix, scalar)
#print(result)


def dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrices dot product.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix or vector.

    Returns:
        np.ndarray: dot product.
    """
    return x @ y 

#приклад використання цієї функції
x = np.array([[1, 2, 3],[4, 5, 6]]) # матриця 2 х 3 
y = np.array([7, 8, 9]) # вектор довжиною 3
#print(dot_product(x,y))


def identity_matrix(dim: int) -> np.ndarray:
    """Create identity matrix with dimension `dim`. 

    Args:
        dim (int): matrix dimension.

    Returns:
        np.ndarray: identity matrix.
    """
    return np.eye(dim) #створює матрицю розміром dim × dim, де на головній діагоналі стоять одиниці

#приклад використання функції
#print(identity_matrix(4))

def matrix_inverse(x: np.ndarray) -> np.ndarray:
    """Compute inverse matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: inverse matrix.
    """
    # перевірка чи матриця є квадратною
    if x.shape[0] != x.shape[1]:
        raise ValueError("Матриця повинна бути квадратною.")

    # перевірка на невиродженість (визначник не дорівнює 0)
    det = np.linalg.det(x)
    if det == 0:
        raise ValueError("Матриця є виродженою (визначник = 0), обернена не існує.")

    return np.linalg.inv(x) # повертає обернену матрицю

# приклад використання функції
x = np.array([[2, 1], [5, 3]])
#print(matrix_inverse(x))


def matrix_transpose(x: np.ndarray) -> np.ndarray:
    """Compute transpose matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: transosed matrix.
    """
    return x.T #повертає транспоновану матрицю

#приклад використання функції
x= np.array([[1, 2, 3],[4, 5, 6]])
#print(matrix_transpose(x))


def hadamard_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute hadamard product.

    Args:
        x (np.ndarray): 1th matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: hadamard produc
    """
    #перевірка чи обидві матриці мають однакову форму
    if x.shape != y.shape:
        raise ValueError("Матриці повинні мати однакові розміри для Адамарового добутку.")
    
    return x * y #перемножує відповідні елементи двох матриць

#приклад використання функції
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
#print(hadamard_product(x, y))


def basis(x: np.ndarray) -> tuple[int]:
    """Compute matrix basis.

    Args:
        x (np.ndarray): matrix.

    Returns:
        tuple[int]: indexes of basis columns.
    """
    raise NotImplementedError


def norm(x: np.ndarray, order: int | float | str) -> float:
    """Matrix norm: Frobenius, Spectral or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 'fro', 2 or inf.

    Returns:
        float: vector norm
    """
    raise NotImplementedError
