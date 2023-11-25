

import numpy as np


def different_arrays():    
    array1 = np.array([1, 1, 2, 3, 5, 8, 13])
    array2 = np.arange(1, 18, 3)
    array3 = np.ones((2, 4)) + 4
    return array1, array2, array3


def matrix_multiplication(matrix_1, matrix_2):
    result = np.dot(matrix_1, matrix_2)
    return result


def multiplication_check(matrix_list):
    for each in range(len(matrix_list) - 1):
        if matrix_list[each].shape[1] != matrix_list[each + 1].shape[0]:
            return False
    return True


def multiply_matrices(matrix_list):
    result = matrix_list[0]
    for each in range(1, len(matrix_list)):
        if result.shape[1] != matrix_list[each].shape[0]:
            return None
        else:
            result = np.dot(result, matrix_list[each])
    return result


def compute_2d_distance(array_1, array_2):
    result = np.sqrt(np.square(array_1[0] - array_2[0]) + np.square(array_1[1] - array_2[1]))
    return result


def compute_multidimensional_distance(array_1, array_2):
    dist = np.linalg.norm(array_1 - array_2)
    return dist


def compute_pair_distances(array_2d):
    x2 = np.sum(array_2d**2, axis=1)
    y2 = x2.copy()
    xy = np.matmul(array_2d, array_2d.T)
    x2 = x2.reshape(-1, 1)
    dist_matrix = np.sqrt(x2 - 2*xy + y2)
    return dist_matrix

