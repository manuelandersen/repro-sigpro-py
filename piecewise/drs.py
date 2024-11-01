import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.io import loadmat

from utilities.estimate_huber_op import estimate_huber_op, create_operators, Operator
from utilities.prox_huberl2_matrix import prox_huber_l2_matrix


# Load data from the .mat file
n = 200
mat_data = loadmat('./data/signal1D_200_v2.mat')
x0 = mat_data['x0'][0]*10
b = x0 + 0.7 * np.random.randn(*x0.shape)

# Create difference operator using the method from estimate_huber_op.py
def create_difference_matrix(n: int) -> np.ndarray:
    """
    Create the first-order discrete difference operator matrix.
    """
    D = np.zeros((n, n))
    np.fill_diagonal(D, 0.5)  # Main diagonal
    np.fill_diagonal(D[:, 1:], -0.5)  # Superdiagonal
    return D

# Create the difference matrix
D = create_difference_matrix(n)

# Calculate norms
D1 = D[::2]  # odd-indexed rows
D2 = D[1::2]  # even-indexed rows
nD1 = np.linalg.norm(D1, ord=2) ** 2
nD2 = np.linalg.norm(D2, ord=2) ** 2

# Setup operators
op, op1, op2 = create_operators(n)

# Calculate diagonal matrices D1 and D2
D1_diag = np.diag(D1 @ D1.T)
D2_diag = np.diag(D2 @ D2.T)

# Parameters
lambda_param = 0.7
rhohub = 0.002

# Calculate optimal step size for DRS
alpha_dr = 1/(1 + 1/rhohub * nD2 * lambda_param)
rho_dr = 1
beta_dr = 1/(1/rhohub * nD1 * lambda_param)

if beta_dr < 4 * alpha_dr/(1 + np.sqrt(1 - alpha_dr * rho_dr)**2):
    gamma_dr = np.sqrt(alpha_dr/rho_dr)
else:
    gamma_dr = np.sqrt(beta_dr/rho_dr)

# Get initial point
_, initgrad = estimate_huber_op(b, np.zeros(len(b)//2), rhohub, op1)

x_init = b + gamma_dr * lambda_param * initgrad

# Function to run DRS iterations
def run_drs_iterations(x_init: np.ndarray, 
                      n_iterations: int,
                      b: np.ndarray,
                      op1: Operator,
                      op2: Operator,
                      D1_diag: np.ndarray,
                      D2_diag: np.ndarray,
                      rhohub: float,
                      gamma_dr: float,
                      lambda_param: float,
                      xbar: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Run DRS iterations and return results and rates.
    """
    xn = x_init.copy()
    rates = []
    
    for _ in range(n_iterations):
        # Store current state for rate calculation if needed
        if xbar is not None:
            rates.append(np.linalg.norm(xn - xbar))
            
        yn = xn - op1.adjoint(1/D1_diag * (op1.direct(xn) - 
             prox_huber_l2_matrix(op1.direct(xn), rhohub, gamma_dr * lambda_param * D1_diag)))
        
        tmp = (2*yn - xn + gamma_dr*b)/(1 + gamma_dr)
        
        zn = tmp - op2.adjoint(1/D2_diag * (op2.direct(tmp) - 
             prox_huber_l2_matrix(op2.direct(tmp), rhohub, 
                                 gamma_dr * lambda_param * D2_diag/(1 + gamma_dr))))
        
        xn = xn + zn - yn
        
    return xn, yn, rates

# Run DRS for maximum iterations to get reference solution
nb_it_max = 500000
xn, _, _ = run_drs_iterations(x_init, nb_it_max, b, op1, op2, D1_diag, D2_diag,
                             rhohub, gamma_dr, lambda_param, None)
xbar = xn
rate0 = np.linalg.norm(x_init - xbar)

# Run DRS for different iteration counts
iterations = [10, 100, 20000]
results = []
rates_list = []

plt.figure(figsize=(15, 5))
hdr = np.array([192, 135, 70])/255  # Color for DRS plot

for i, n_iter in enumerate(iterations, 1):
    xn, yn, rates = run_drs_iterations(x_init, n_iter, b, op1, op2, D1_diag, D2_diag,
                                     rhohub, gamma_dr, lambda_param, xbar)
    results.append(yn)
    rates_list.append(rates)
    
    plt.subplot(1, 3, i)
    plt.plot(b, color='0.8', label='Noisy')
    plt.plot(x0, 'k', label='Original')
    plt.plot(yn, color=hdr, linewidth=1.2, label='DRS')
    plt.grid(True)
    plt.title(f'{n_iter} iterations')
    if i == 1:
        plt.legend()

plt.tight_layout()

# Calculate theoretical rate
rDR = min(1/2 + 1/2 * max((1 - gamma_dr * rho_dr)/(1 + gamma_dr * rho_dr),
                          (gamma_dr/alpha_dr - 1)/(gamma_dr/alpha_dr + 1)),
          (beta_dr + gamma_dr**2 * rho_dr)/(beta_dr + gamma_dr * beta_dr * rho_dr + gamma_dr**2 * rho_dr))

# Plot convergence rates
plt.figure()
it = np.arange(len(rates_list[2]))
plt.semilogy(rates_list[2], color=hdr, linewidth=1.2, label='Experimental DRS')
plt.semilogy(rDR**it * rate0, color=hdr, linewidth=2, linestyle='-.', label='Theoretical DRS')
plt.xlim([0, 6000])
plt.ylim([1e-12, 2])
plt.legend()
plt.grid(True)
plt.title('Convergence Rate')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()