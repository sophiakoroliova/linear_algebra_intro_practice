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

""" 
Вивід програми : 

Вхідний вектор: [ 1 -2  3  0]
Негативний вектор: [-1  2 -3  0]
Вхідна матриця:
 [[ 10 -20]
 [ 30 -40]]
Негативна матриця:
 [[-10  20]
 [-30  40]]
"""

def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the input vector or matrix with the order of elements reversed.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with the order of elements reversed.
    """
    # використовуємо вбудовану функцію np.flip(x) що обертає масив вздовж усіх осей
    # для 1D-вектора функція змінює порядок елементів з кінця на початок 
    # для 2-D-матриці функція обертає матрицю на 180 градусів (і рядки, і стовпці)
    return np.flip(x) 

# Приклад з вектором
vector = np.array([1, -2, 3, 0])
reversed_vector = reverse_matrix(vector)
print("Вхідний вектор:", vector)
print("Обернений вектор:", reversed_vector)

# Приклад з матрицею
matrix = np.array([
    [10, -20],
    [30, -40]
])
reversed_matrix = reverse_matrix(matrix)
print("Вхідна матриця:\n", matrix)
print("Обернена матриця:\n", reversed_matrix)

"""
Вивід програми : 

Вхідний вектор: [ 1 -2  3  0]
Обернений вектор: [ 0  3 -2  1]
Вхідна матриця:
 [[ 10 -20]
 [ 30 -40]]
Обернена матриця:
 [[-40  30]
 [-20  10]]
"""

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
    alpha = np.deg2rad(alpha_deg) # конвертуємо кут у радіани

    # матриця масштабування
    S = np.array([
        [scale[0], 0],
        [0, scale[1]]
    ])

    # матриця зсуву
    Sh = np.array([
        [1, np.tan(np.deg2rad(shear[0]))],
        [np.tan(np.deg2rad(shear[1])), 1]
    ])

    # матриця обертання
    R = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha),  np.cos(alpha)]
    ])

    # повна афінна матриця (без зсуву)
    A = R @ Sh @ S

    # додаємо трансляцію (зсув)
    t = np.array(translate)

    # перетворюємо точки
    if x.ndim == 1:
        x_transformed = A @ x + t # якщо це один вектор
    else:
        x_transformed = (A @ x.T).T + t # якщо це кілька точок (n×2)

    return x_transformed

points = np.array([
    [0, 0],
    [1, 0],
    [0, 1]
])

transformed = affine_transform(
    points,
    alpha_deg=45,
    scale=(1.2, 0.8),
    shear=(10, 5),
    translate=(2, 3)
)

print(transformed)

"""
Вивід програми : 

[[2.         1.        ]
 [2.70710678 1.70710678]
 [1.29289322 1.70710678]]

"""
