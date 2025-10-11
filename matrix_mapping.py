import numpy as np

def negative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the negation of each element in the input vector or matrix.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with each element negated.
    """
    # застосовуємо унарний оператор від'ємного числа до всього масиву (NumPy виконує це поелементно)
    return -x

# Приклад з вектором
vector = np.array([1, -2, 3, 0])
neg_vector = negative_matrix(vector)
print("Вхідний вектор:", vector)
print("Негативний вектор:", neg_vector)

# Приклад з матрицею
matrix = np.array([
    [10, -20],
    [30, -40]
])
neg_matrix = negative_matrix(matrix)
print("Вхідна матриця:\n", matrix)
print("Негативна матриця:\n", neg_matrix)


def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the input vector or matrix with the order of elements reversed.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with the order of elements reversed.
    """
    raise NotImplementedError


def affine_transform(
    x: np.ndarray, alpha_deg: float, scale: tuple[float, float], shear: tuple[float, float],
    translate: tuple[float, float],
) -> np.ndarray:
    """Compute affine transformation

    Args:
        x (np.ndarray): vector n*1 or matrix n*n.
        alpha_deg (float): rotation angle in deg.
        scale (tuple[float, float]): x, y scale factor.
        shear (tuple[float, float]): x, y shear factor.
        translate (tuple[float, float]): x, y translation factor.

    Returns:
        np.ndarray: transformed matrix.
    """
    raise NotImplementedError
