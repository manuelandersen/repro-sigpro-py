import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Literal
from scipy.io import loadmat

from utilities.estimate_huber_op import estimate_huber_op, create_operators, Operator
from utilities.estimate_l2 import estimate_l2

def run_gradient_iterations(
    x_init: np.ndarray,
    n_iterations: int,
    b: np.ndarray,
    op: Operator,
    gamma_grad: float,
    lambda_param: float,
    rhohub: float,
    method: Literal["banach-picard", "krasnoselskii-mann"] = "banach-picard",
    alpha: float = 0.5,  # Krasnoselskii-Mann parameter
    A: float = 1
) -> Tuple[np.ndarray, List[float]]:
    """
    Run gradient descent iterations with either Banach-Picard or Krasnoselskii-Mann method.
    
    Args:
        x_init: Initial point
        n_iterations: Number of iterations
        b: Target signal
        op: Operator containing direct and adjoint operations
        gamma_grad: Step size
        lambda_param: Lambda parameter
        rhohub: Huber parameter
        method: "banach-picard" or "krasnoselskii-mann"
        alpha: Relaxation parameter for Krasnoselskii-Mann (in (0,1])
        A: Identity operator (default=1)
        
    Returns:
        Tuple containing:
        - Final solution
        - List of criterion values at each iteration
    """
    zn = x_init.copy()
    crits = []
    
    for _ in range(n_iterations):
        # Calculate L2 criterion and gradient
        crit1, grad1 = estimate_l2(zn.reshape(-1, 1), b, A)
        
        # Calculate Huber criterion and gradient
        crit2, grad2 = estimate_huber_op(zn, np.zeros_like(zn), rhohub, op)
        
        # Store total criterion
        crits.append(float(crit1 + lambda_param * crit2))
        
        # Calculate gradient step
        grad_step = gamma_grad * (grad1.flatten() + lambda_param * grad2)
        
        # Update step based on chosen method
        if method == "banach-picard":
            zn = zn - grad_step
        else:  # Krasnoselskii-Mann
            zn = zn - alpha * grad_step
        
    return zn, crits

# Load data
n = 200
mat_data = loadmat('./data/signal1D_200_v2.mat')
x0 = mat_data['x0'][0] * 10
b = x0 + 0.7 * np.random.randn(*x0.shape)

# Create operators
op, _, _ = create_operators(n)

# Calculate norm of difference matrix
D = op.direct(np.eye(n))
nD = np.linalg.norm(D, ord=2) ** 2

# Algorithm parameters
A = 1
lambda_param = 0.7
rhohub = 0.002

# Calculate gradient descent parameters
alpha_grad = 1
rho_grad = 1
beta_grad = 1/(1/rhohub * nD * lambda_param)
gamma_grad = 2/(rho_grad + 1/alpha_grad + 1/beta_grad)

# Define colors for plots
hgrad_bp = np.array([76, 0, 153])/255  # Color for Banach-Picard
hgrad_km = np.array([153, 76, 0])/255  # Color for Krasnoselskii-Mann

# Run gradient descent for different iteration counts
plt.figure(figsize=(15, 5))
iterations = [10, 100, 20000]
final_crits_bp = None
final_crits_km = None

for i, n_iter in enumerate(iterations, 1):
    print(f'Running gradient descent variants for {n_iter} iterations...')
    
    # Run both variants
    yn_grad_bp, crits_bp = run_gradient_iterations(
        b, n_iter, b, op, gamma_grad, lambda_param, rhohub, 
        method="banach-picard"
    )
    
    yn_grad_km, crits_km = run_gradient_iterations(
        b, n_iter, b, op, gamma_grad, lambda_param, rhohub,
        method="krasnoselskii-mann", alpha=0.5
    )
    
    # Store criterions from longest run for convergence plot
    if n_iter == max(iterations):
        final_crits_bp = crits_bp
        final_crits_km = crits_km
    
    # Plot results
    plt.subplot(1, 3, i)
    plt.plot(b, color='0.8', label='Noisy')
    plt.plot(x0, 'k', label='Original')
    plt.plot(yn_grad_bp, color=hgrad_bp, linewidth=1.2, label='EA (BP)')
    plt.plot(yn_grad_km, color=hgrad_km, linewidth=1.2, label='EA (KM)')
    plt.grid(True)
    plt.title(f'{n_iter} iterations')
    if i == 1:
        plt.legend(fontsize=12)

plt.tight_layout()

# Plot convergence rates
plt.figure(figsize=(10, 6))
tau = gamma_grad

# Calculate theoretical rates
rG_bp = max(abs(1 - tau * rho_grad), 
            abs(1 - tau * (1/beta_grad + 1/alpha_grad)))

# Krasnoselskii-Mann rate (modified by alpha parameter)
alpha_km = 0.5  # Same as used in iterations
rG_km = max(abs(1 - alpha_km * tau * rho_grad), 
            abs(1 - alpha_km * tau * (1/beta_grad + 1/alpha_grad)))

# Mask values where condition isn't met
if tau > 2 * alpha_grad * beta_grad / (beta_grad + alpha_grad):
    rG_bp = np.nan
    rG_km = np.nan

# Plot experimental criterion values
plt.semilogy(final_crits_bp, color=hgrad_bp, linewidth=2, label='Experimental EA (BP)')
plt.semilogy(final_crits_km, color=hgrad_km, linewidth=2, label='Experimental EA (KM)')

# Plot theoretical rates
it = np.arange(len(final_crits_bp))
plt.semilogy(rG_bp**it * final_crits_bp[0], color=hgrad_bp, linewidth=4, 
            linestyle='--', label='Theoretical EA (BP)')
plt.semilogy(rG_km**it * final_crits_km[0], color=hgrad_km, linewidth=4, 
            linestyle='--', label='Theoretical EA (KM)')

plt.xlim([0, 6000])
plt.ylim([1.79329634085052e-12, 2])
plt.legend(fontsize=12)
plt.grid(True)
plt.title('Convergence Rate Comparison')
plt.xlabel('Iterations')
plt.ylabel('Criterion Value')

plt.show()