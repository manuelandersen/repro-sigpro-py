import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.io import loadmat

from utilities.estimate_huber_op import estimate_huber_op, create_operators, Operator
from utilities.prox_huberl2_matrix import prox_huber_l2_matrix
from utilities.estimate_l2 import estimate_l2
from utilities.prox_l2diff import prox_l2diff

# Set up colors for different FBS variants
hfb2 = np.array([153, 0, 153])/255    # FBS2 color
hfb3a = np.array([218, 152, 207])/255  # FBS3a color
hfb3b = np.array([255, 0, 127])/255    # FBS3b color

# Load data from the .mat file
n = 200
mat_data = loadmat('./data/signal1D_200_v2.mat')
x0 = mat_data['x0'][0]*10
b = x0 + 0.7 * np.random.randn(*x0.shape)

# Create operators using the existing create_operators function
op, op1, op2 = create_operators(n)

# Calculate norms using the difference matrices
D = op.direct(np.eye(n))  # Get the full difference matrix
nD = np.linalg.norm(D, ord=2) ** 2
nD1 = np.linalg.norm(D[::2], ord=2) ** 2  # odd-indexed rows
nD2 = np.linalg.norm(D[1::2], ord=2) ** 2  # even-indexed rows

# Calculate diagonal matrices D1 and D2
D1_diag = np.diag(D[::2] @ D[::2].T)
D2_diag = np.diag(D[1::2] @ D[1::2].T)

# Algorithm parameters
A = 1
lambda_param = 0.7
rhohub = 0.002
nb_it_max = 500000

# Calculate FBS parameters
alpha_fb2 = 1/(1 + 1/rhohub * nD2 * lambda_param)
rho_fb2 = 1
beta_fb2 = 1/(1/rhohub * nD1 * lambda_param)
gamma_fb2 = 2/(rho_fb2 + 1/alpha_fb2)

def run_fbs_iterations(variant: str,
                      x_init: np.ndarray,
                      n_iterations: int,
                      b: np.ndarray,
                      op: Operator,
                      op1: Operator,
                      op2: Operator,
                      D1_diag: np.ndarray,
                      D2_diag: np.ndarray,
                      params: dict,
                      xbar: np.ndarray = None) -> Tuple[np.ndarray, List[float]]:
    """
    Run FBS iterations for different variants.
    """
    xn = x_init.copy()
    rates = []
    
    for _ in range(n_iterations):
        if xbar is not None:
            rates.append(np.linalg.norm(xn - xbar))
            
        if variant == "FBS2":
            # Use zeros with correct size for op2 (half the signal length)
            _, grad1 = estimate_huber_op(xn, np.zeros(n//2), rhohub, op2)
            _, grad3 = estimate_l2(xn, b, A)
            tmp = xn - params['gamma'] * (lambda_param * grad1 + grad3)
            # Use zeros with correct size for op1
            xn = tmp - op1.adjoint(1/D1_diag * (op1.direct(tmp) - 
                 prox_huber_l2_matrix(op1.direct(tmp), rhohub, 
                                    params['gamma'] * lambda_param * D1_diag)))
            
        elif variant == "FBS3a":
            # Use zeros with the full signal length for op
            _, grad2 = estimate_huber_op(xn, np.zeros(n), rhohub, op)
            xn = prox_l2diff(xn - params['gamma'] * lambda_param * grad2,
                           b, A, params['gamma'], params['Minv'])
            
        elif variant == "FBS3b":
            # Use zeros with correct size for op1
            _, grad1 = estimate_huber_op(xn, np.zeros(n//2), rhohub, op1)
            tmp = (xn - params['gamma'] * lambda_param * grad1 + 
                  params['gamma'] * b)/(1 + params['gamma'])
            xn = tmp - op2.adjoint(1/D2_diag * (op2.direct(tmp) - 
                 prox_huber_l2_matrix(op2.direct(tmp), rhohub,
                                    params['gamma'] * lambda_param * 
                                    D2_diag/(1 + params['gamma']))))
            
    return xn, rates

# Get reference solution using FBS2 with many iterations
params_fb2 = {'gamma': gamma_fb2}
xn_ref, _ = run_fbs_iterations("FBS2", b, nb_it_max, b, op, op1, op2,
                              D1_diag, D2_diag, params_fb2)
ybar = xn_ref
rate0 = np.linalg.norm(b - ybar)

# Set up parameters for all variants
alpha_fb3a = 1
rho_fb3a = 1
beta_fb3a = 1/(1/rhohub * nD * lambda_param)
gamma_fb3a = 1.99 * beta_fb3a
Minv_fb3a = 1/(1 + gamma_fb3a)

alpha_fb3b = 1/(1 + 1/rhohub * nD2 * lambda_param)
rho_fb3b = 1
beta_fb3b = 1/(1/rhohub * nD1 * lambda_param)
gamma_fb3b = 2 * beta_fb3b

params = {
    "FBS2": {'gamma': gamma_fb2},
    "FBS3a": {'gamma': gamma_fb3a, 'Minv': Minv_fb3a},
    "FBS3b": {'gamma': gamma_fb3b}
}

# Plot results for different iteration counts
plt.figure(figsize=(15, 5))
variants = ["FBS2", "FBS3a", "FBS3b"]
colors = {'FBS2': hfb2, 'FBS3a': hfb3a, 'FBS3b': hfb3b}
styles = {'FBS2': '-', 'FBS3a': '--', 'FBS3b': '--'}

iterations = [10, 100, 20000]
all_rates = {variant: [] for variant in variants}

for i, n_iter in enumerate(iterations, 1):
    plt.subplot(1, 3, i)
    plt.plot(b, color='0.8', label='Noisy')
    plt.plot(x0, 'k', label='Original')
    
    for variant in variants:
        xn, rates = run_fbs_iterations(variant, b, n_iter, b, op, op1, op2,
                                     D1_diag, D2_diag, params[variant], ybar)
        if n_iter == max(iterations):
            all_rates[variant] = rates
            
        plt.plot(xn, color=colors[variant], linewidth=1.2,
                linestyle=styles[variant], label=variant)
    
    plt.grid(True)
    plt.title(f'{n_iter} iterations')
    if i == 1:
        plt.legend()

plt.tight_layout()

# Calculate theoretical rates and plot convergence
plt.figure(figsize=(10, 6))

# FBS2
tau = gamma_fb2
r_FB2 = max(abs(1 - tau * rho_fb2), abs(1 - tau/alpha_fb2))
it = np.arange(len(all_rates['FBS2']))
plt.semilogy(all_rates['FBS2'], color=hfb2, linewidth=1.2, label='FBS2 Exp')
plt.semilogy(r_FB2**it * rate0, color=hfb2, linewidth=2,
            linestyle='-.', label='FBS2 Theo')

# FBS3a
tau = gamma_fb3a
r_FB3a = 1/(1 + tau * rho_fb3a)
plt.semilogy(all_rates['FBS3a'], color=hfb3a, linewidth=1.2, label='FBS3a Exp')
plt.semilogy(r_FB3a**it * rate0, color=hfb3a, linewidth=2,
            linestyle='-.', label='FBS3a Theo')

# FBS3b
tau = gamma_fb3b
r_FB3b = 1/(1 + tau * rho_fb3b)
plt.semilogy(all_rates['FBS3b'], color=hfb3b, linewidth=1.2, label='FBS3b Exp')
plt.semilogy(r_FB3b**it * rate0, color=hfb3b, linewidth=2,
            linestyle='-.', label='FBS3b Theo')

plt.xlim([0, 6000])
plt.ylim([1e-12, 2])
plt.legend(fontsize=12)
plt.grid(True)
plt.title('FBS Variants Convergence Rates')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()