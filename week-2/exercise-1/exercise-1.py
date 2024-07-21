import numpy as np

# 1. Độ dài của vector


def compute_vector_length(vector):
    if type(vector) != np.ndarray:
        raise ValueError("Input must be an array")
    if vector.ndim != 1:
        raise ValueError("Input must be a 1D vector")
    len_of_vector = np.sqrt(vector@vector.T)
    return len_of_vector


# 2. Phép tích vô hướng

def compute_dot_product(vector_1, vector_2):
    if type(vector_1) != np.ndarray or type(vector_2) != np.ndarray:
        raise ValueError("Input must be an array")
    if vector_1.ndim != 1 or vector_2.ndim != 1:
        raise ValueError("Input must be a 1D vector")
    if vector_1.shape != vector_2.shape:
        raise ValueError("Input must have the same shape")
    dot_product = vector_1@vector_2
    return dot_product

# 3. Nhân vector với ma trận


def matrix_multi_vector(matrix, vector):
    if type(matrix) != np.ndarray or type(vector) != np.ndarray:
        raise ValueError("Input must be numpy array")
    if matrix.ndim != 2 or vector.ndim != 1:
        raise ValueError("Invalid input shape")
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError("Incompatible matrix and vector shapes")
    matrix_multi_vector = matrix@vector
    return matrix_multi_vector

# 4. Nhân ma trận với ma trận


def matrix_multi_matrix(matrix_1, matrix_2):
    if type(matrix_1) != np.ndarray or type(matrix_2) != np.ndarray:
        raise ValueError("Input must be a numpy array")
    if matrix_1.ndim != 2 or matrix_2.ndim != 2:
        raise ValueError("Invalid input shape")
    if matrix_1.shape[1] != matrix_2.shape[0]:
        raise ValueError("Incompatible matrix shapes")
    matrix_multi_matrix = np.einsum("ij,jk->ik", matrix_1, matrix_2)
    return matrix_multi_matrix

# 5. Ma trận nghịch đảo:


def inverse_matrix(matrix):
    if type(matrix) != np.ndarray:
        raise ValueError("Input must be a numpy array")
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix")
    if matrix.shape[0] != 2:
        raise ValueError("Input must be a 2x2 matrix")
    det = np.linalg.det(matrix)
    if det == 0:
        raise ValueError("Non-Invertible matrix")
    inverse_matrix = 1/det * \
        np.array([[matrix[1, 1], -matrix[0, 1]],
                 [-matrix[1, 0], matrix[0, 0]]])
    return inverse_matrix


if __name__ == "__main__":
    print('#################Length of vector#################')
    vector = np.array([1, 2, 3])
    print(f"Input vector: {vector}")
    print(compute_vector_length(vector))
    print('#################Dot product#################')
    vector_1 = np.array([1, 2, 3])
    vector_2 = np.array([4, 5, 6])
    print(f"Vector 1: {vector_1}")
    print(f"Vector 2: {vector_2}")
    print(compute_dot_product(vector_1, vector_2))
    print('#################Matrix multiplication#################')
    matrix = np.array([[1, 2], [3, 4]])
    vector = np.array([5, 6])
    print(f"Matrix: {matrix}")
    print(f"Vector: {vector}")
    print(matrix_multi_vector(matrix, vector))
    print('#################Matrix multiplication#################')
    matrix_1 = np.array([[1, 2], [3, 4]])
    matrix_2 = np.array([[5, 6], [7, 8]])
    print(f"Matrix 1: {matrix_1}")
    print(f"Matrix 2: {matrix_2}")
    print(matrix_multi_matrix(matrix_1, matrix_2))
    print('#################Inverse matrix#################')
    matrix = np.array([[1, 2], [3, 4]])
    print(f"Matrix: {matrix}")
    print(inverse_matrix(matrix))
