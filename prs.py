import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.io import loadmat

from utilities.estimate_huber_op import estimate_huber_op, create_operators, Operator
from utilities.prox_huberl2_matrix import prox_huber_l2_matrix
from utilities.function_huber_l2 import function_huber_l2

# Load data from the .mat file
n = 200
mat_data = loadmat('./data/signal1D_200_v2.mat')
x0 = mat_data['x0'][0]*10
b = x0 + 0.7 * np.random.randn(*x0.shape)

# Create operators using the existing create_operators function
op, op1, op2 = create_operators(n)

# Calculate norms using the difference matrices
D = op.direct(np.eye(n))  # Get the full difference matrix
D1 = D[::2]  # odd-indexed rows
D2 = D[1::2]  # even-indexed rows
nD = np.linalg.norm(D, ord=2) ** 2
nD1 = np.linalg.norm(D1, ord=2) ** 2
nD2 = np.linalg.norm(D2, ord=2) ** 2

# Calculate diagonal matrices D1 and D2
D1_diag = np.diag(D1 @ D1.T)
D2_diag = np.diag(D2 @ D2.T)

# Algorithm parameters
A = 1  # Identity operator
lambda_param = 0.7
rhohub = 0.002
nb_it_max = 500000

# Calculate PRS parameters
alpha_pr = 1/(1 + 1/rhohub * nD2 * lambda_param)
rho_pr = 1
beta_pr = 1/(1/rhohub * nD1 * lambda_param)
gamma_pr = np.sqrt(alpha_pr/rho_pr)

def run_prs_iterations(x_init: np.ndarray, 
                      n_iterations: int,
                      b: np.ndarray,
                      op: Operator,
                      op1: Operator,
                      op2: Operator,
                      D1_diag: np.ndarray,
                      D2_diag: np.ndarray,
                      rhohub: float,
                      gamma_pr: float,
                      lambda_param: float,
                      xbar: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    """
    Run PRS iterations and return results.
    """
    xn = x_init.copy()
    rates = []
    crits = []
    
    for _ in range(n_iterations):
        yn = xn - op1.adjoint(1/D1_diag * (op1.direct(xn) - 
             prox_huber_l2_matrix(op1.direct(xn), rhohub, gamma_pr * lambda_param * D1_diag)))
        
        tmp = (2*yn - xn + gamma_pr*b)/(1 + gamma_pr)
        
        zn = tmp - op2.adjoint(1/D2_diag * (op2.direct(tmp) - 
             prox_huber_l2_matrix(op2.direct(tmp), rhohub, 
                                 gamma_pr * lambda_param * D2_diag/(1 + gamma_pr))))
        
        xn = 2*zn - (2*yn - xn)
        
        # Calculate rate and criterion if needed
        if xbar is not None:
            rates.append(np.linalg.norm(xn - xbar))
            crits.append(0.5 * np.sum((A*yn - b)**2) + 
                        lambda_param * function_huber_l2(op.direct(yn), rhohub))
            
    return xn, yn, rates, crits

# Get initial point
_, initgrad = estimate_huber_op(b, np.zeros(len(b)//2), rhohub, op1)
x_init = b + gamma_pr * lambda_param * initgrad

# Run PRS for maximum iterations to get reference solution
xn, yn_bar, _, crit_prbar = run_prs_iterations(x_init, nb_it_max, b, op, op1, op2, 
                                              D1_diag, D2_diag, rhohub, gamma_pr, 
                                              lambda_param)
xbar_pr = xn
rate0_pr = np.linalg.norm(x_init - xbar_pr)

# Plot results for different iteration counts
plt.figure(figsize=(15, 5))
hdr = np.array([100, 29, 29])/255  # Color for PRS plot

# Run PRS for different numbers of iterations
iterations = [10, 100, 20000]
all_rates = []

for i, n_iter in enumerate(iterations, 1):
    # Run PRS
    _, initgrad = estimate_huber_op(b, np.zeros(len(b)//2), rhohub, op1)
    x_init = b + gamma_pr * lambda_param * initgrad
    
    xn, yn, rates, crits = run_prs_iterations(x_init, n_iter, b, op, op1, op2, 
                                             D1_diag, D2_diag, rhohub, gamma_pr, 
                                             lambda_param, xbar_pr)
    
    if n_iter == max(iterations):
        final_rates = rates
    
    plt.subplot(1, 3, i)
    plt.plot(b, color='0.8', label='Noisy')
    plt.plot(x0, 'k', label='Original')
    plt.plot(yn, color=hdr, linewidth=1.2, label='PRS')
    plt.grid(True)
    plt.title(f'{n_iter} iterations')
    if i == 1:
        plt.legend()

plt.tight_layout()

# Calculate theoretical rate
tau = gamma_pr
r_PR = max((1 - tau * rho_pr)/(1 + tau * rho_pr),
           (tau/alpha_pr - 1)/(tau/alpha_pr + 1))

# Plot convergence rates
plt.figure()
it = np.arange(len(final_rates))
plt.semilogy(final_rates, color=hdr, linewidth=1.2, label='Experimental PRS')
plt.semilogy(r_PR**it * rate0_pr, color=hdr, linewidth=2, linestyle='-.', label='Theoretical PRS')
plt.xlim([0, 6000])
plt.ylim([1e-12, 2])
plt.legend(fontsize=16)
plt.grid(True)
plt.title('Convergence Rate')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()