import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Literal
from scipy.io import loadmat

from utilities.estimate_huber_op import estimate_huber_op, create_operators, Operator
from utilities.prox_huberl2_matrix import prox_huber_l2_matrix
from utilities.estimate_l2 import estimate_l2
from utilities.prox_l2diff import prox_l2diff

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
                      method: Literal["banach-picard", "krasnoselskii-mann"] = "banach-picard",
                      alpha: float = 0.5,  # Krasnoselskii-Mann parameter
                      xbar: np.ndarray = None) -> Tuple[np.ndarray, List[float]]:
    """
    Run FBS iterations for different variants with Banach-Picard or Krasnoselskii-Mann method.
    """
    xn = x_init.copy()
    rates = []
    
    for _ in range(n_iterations):
        if xbar is not None:
            rates.append(np.linalg.norm(xn - xbar))
        
        # Store current point for KM iteration
        xn_current = xn.copy()
            
        if variant == "FBS2":
            # Use zeros with correct size for op2 (half the signal length)
            _, grad1 = estimate_huber_op(xn, np.zeros(params['n']//2), params['rhohub'], op2)
            _, grad3 = estimate_l2(xn.reshape(-1, 1), b, params['A'])
            tmp = xn - params['gamma'] * (params['lambda_param'] * grad1 + grad3.flatten())
            xn_next = tmp - op1.adjoint(1/D1_diag * (op1.direct(tmp) - 
                     prox_huber_l2_matrix(op1.direct(tmp), params['rhohub'], 
                                        params['gamma'] * params['lambda_param'] * D1_diag)))
            
        elif variant == "FBS3a":
            _, grad2 = estimate_huber_op(xn, np.zeros(params['n']), params['rhohub'], op)
            xn_next = prox_l2diff(xn - params['gamma'] * params['lambda_param'] * grad2,
                               b, params['A'], params['gamma'], params['Minv'])
            
        elif variant == "FBS3b":
            _, grad1 = estimate_huber_op(xn, np.zeros(params['n']//2), params['rhohub'], op1)
            tmp = (xn - params['gamma'] * params['lambda_param'] * grad1 + 
                  params['gamma'] * b)/(1 + params['gamma'])
            xn_next = tmp - op2.adjoint(1/D2_diag * (op2.direct(tmp) - 
                     prox_huber_l2_matrix(op2.direct(tmp), params['rhohub'],
                                        params['gamma'] * params['lambda_param'] * 
                                        D2_diag/(1 + params['gamma']))))
        
        # Apply either Banach-Picard or Krasnoselskii-Mann update
        if method == "banach-picard":
            xn = xn_next
        else:  # Krasnoselskii-Mann
            xn = xn_current + alpha * (xn_next - xn_current)
            
    return xn, rates

# Set up distinct color schemes for Banach-Picard and Krasnoselskii-Mann
# Banach-Picard: Purple/Blue spectrum
colors_bp = {
    "FBS2": np.array([75, 0, 130])/255,     # Dark Purple
    "FBS3a": np.array([0, 0, 205])/255,     # Dark Blue
    "FBS3b": np.array([138, 43, 226])/255   # Purple
}

# Krasnoselskii-Mann: Orange/Red spectrum
colors_km = {
    "FBS2": np.array([255, 69, 0])/255,     # Red-Orange
    "FBS3a": np.array([205, 92, 92])/255,   # Indian Red
    "FBS3b": np.array([178, 34, 34])/255    # Firebrick
}

# Load data
n = 200
mat_data = loadmat('./data/signal1D_200_v2.mat')
x0 = mat_data['x0'][0]*10
b = x0 + 0.7 * np.random.randn(*x0.shape)

# Create operators
op, op1, op2 = create_operators(n)

# Calculate norms using the difference matrices
D = op.direct(np.eye(n))
nD = np.linalg.norm(D, ord=2) ** 2
nD1 = np.linalg.norm(D[::2], ord=2) ** 2
nD2 = np.linalg.norm(D[1::2], ord=2) ** 2

# Calculate diagonal matrices
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

alpha_fb3a = 1
rho_fb3a = 1
beta_fb3a = 1/(1/rhohub * nD * lambda_param)
gamma_fb3a = 1.99 * beta_fb3a
Minv_fb3a = 1/(1 + gamma_fb3a)

alpha_fb3b = 1/(1 + 1/rhohub * nD2 * lambda_param)
rho_fb3b = 1
beta_fb3b = 1/(1/rhohub * nD1 * lambda_param)
gamma_fb3b = 2 * beta_fb3b

# Set up parameters for all variants
params = {
    "FBS2": {'gamma': gamma_fb2, 'A': A, 'lambda_param': lambda_param, 
             'rhohub': rhohub, 'n': n},
    "FBS3a": {'gamma': gamma_fb3a, 'Minv': Minv_fb3a, 'A': A, 
              'lambda_param': lambda_param, 'rhohub': rhohub, 'n': n},
    "FBS3b": {'gamma': gamma_fb3b, 'A': A, 'lambda_param': lambda_param, 
              'rhohub': rhohub, 'n': n}
}

# Get reference solutions (using Banach-Picard for reference)
xn_ref = {}
for variant in ["FBS2", "FBS3a", "FBS3b"]:
    xn_ref[variant], _ = run_fbs_iterations(variant, b, nb_it_max, b, op, op1, op2,
                                          D1_diag, D2_diag, params[variant])

# Plot results for different iteration counts
iterations = [10, 100, 20000]
plt.figure(figsize=(15, 5))
all_rates = {}

for i, n_iter in enumerate(iterations, 1):
    plt.subplot(1, 3, i)
    plt.plot(b, color='0.8', label='Noisy')
    plt.plot(x0, 'k', label='Original')
    
    for variant in ["FBS2", "FBS3a", "FBS3b"]:
        # Run Banach-Picard
        xn_bp, rates_bp = run_fbs_iterations(
            variant, b, n_iter, b, op, op1, op2, D1_diag, D2_diag,
            params[variant], method="banach-picard", xbar=xn_ref[variant]
        )
        
        # Run Krasnoselskii-Mann
        xn_km, rates_km = run_fbs_iterations(
            variant, b, n_iter, b, op, op1, op2, D1_diag, D2_diag,
            params[variant], method="krasnoselskii-mann", alpha=0.241, xbar=xn_ref[variant]
        )
        
        if n_iter == max(iterations):
            all_rates[(variant, "banach-picard")] = rates_bp
            all_rates[(variant, "krasnoselskii-mann")] = rates_km
        
        plt.plot(xn_bp, color=colors_bp[variant], linewidth=1.2, 
                label=f"{variant} (BP)")
        plt.plot(xn_km, color=colors_km[variant], linewidth=1.2, 
                linestyle='--', label=f"{variant} (KM)")
    
    plt.grid(True)
    plt.title(f'{n_iter} iterations')
    if i == 1:
        plt.legend(fontsize=8)

plt.tight_layout()

# Plot convergence rates
plt.figure(figsize=(12, 8))

# Calculate theoretical rates and plot convergence
for variant in ["FBS2", "FBS3a", "FBS3b"]:
    if variant == "FBS2":
        tau = gamma_fb2
        r_BP = max(abs(1 - tau * rho_fb2), abs(1 - tau/alpha_fb2))
        r_KM = max(abs(1 - 0.5 * tau * rho_fb2), abs(1 - 0.5 * tau/alpha_fb2))
    elif variant == "FBS3a":
        tau = gamma_fb3a
        r_BP = 1/(1 + tau * rho_fb3a)
        r_KM = 1/(1 + 0.5 * tau * rho_fb3a)
    else:  # FBS3b
        tau = gamma_fb3b
        r_BP = 1/(1 + tau * rho_fb3b)
        r_KM = 1/(1 + 0.5 * tau * rho_fb3b)
    
    # Plot Banach-Picard
    rates_bp = all_rates[(variant, "banach-picard")]
    plt.semilogy(rates_bp, color=colors_bp[variant], linewidth=1.2,
                label=f"{variant} (BP) Exp")
    it = np.arange(len(rates_bp))
    plt.semilogy(r_BP**it * rates_bp[0], color=colors_bp[variant], 
                linewidth=2, linestyle='--', label=f"{variant} (BP) Theo")
    
    # Plot Krasnoselskii-Mann
    rates_km = all_rates[(variant, "krasnoselskii-mann")]
    plt.semilogy(rates_km, color=colors_km[variant], linewidth=1.2,
                label=f"{variant} (KM) Exp")
    plt.semilogy(r_KM**it * rates_km[0], color=colors_km[variant], 
                linewidth=2, linestyle='--', label=f"{variant} (KM) Theo")

plt.xlim([0, 6000])
plt.ylim([1e-12, 2])
plt.legend(fontsize=8, ncol=2)
plt.grid(True)
plt.title('FBS Variants Convergence Rates\nBanach-Picard (Purple/Blue) vs Krasnoselskii-Mann (Orange/Red)')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()