import numpy as np
from utilities.prox_huberl2_matrix import prox_huber_l2_matrix

def test_vector_input():
    x = np.array([-3, -2, -1, 0, 1, 2, 3])
    rho = 1.0
    gamma = 0.5
    result = prox_huber_l2_matrix(x, rho, gamma)
    expected = np.array([-2.5, -1.5, -0.66666667, 0, 0.66666667, 1.5, 2.5])
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_matrix_inout():
    x = np.array([[-2, 1, 3], [-1, 2, 4], [0, -3, -2]])
    rho = 1.0
    gamma = 0.5
    result = prox_huber_l2_matrix(x, rho, gamma)
    expected = np.array([[-1.5, 0.66666667, 2.5], [-0.66666667, 1.5, 3.5], [0, -2.5, -1.5]])
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_vector_input_with_vector_gamma():
    x = np.array([-3, -2, -1, 0, 1, 2, 3])
    rho = 1.0
    gamma = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    result = prox_huber_l2_matrix(x, rho, gamma)
    expected = np.array([-2.5, -1.4, -0.58823529, 0, 0.52631579, 1.0, 1.9])
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_near_threshold():
    x = np.array([-1.5, -0.8, 0.8, 1.5])
    rho = 1.0
    gamma = 0.5
    result = prox_huber_l2_matrix(x, rho, gamma)
    print(result)
    expected = np.array([-1.0, -0.53333333, 0.53333333, 1.0])
    print(expected)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_zero_input():
    x = np.zeros((3, 3))
    rho = 1.0
    gamma = 0.5
    result = prox_huber_l2_matrix(x, rho, gamma)
    expected = np.zeros((3, 3))
    np.testing.assert_array_equal(result, expected)