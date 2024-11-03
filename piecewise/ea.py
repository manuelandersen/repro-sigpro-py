import sys
import os
sys.path.append(os.path.abspath('../utilities'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.io import loadmat
from datetime import datetime

from estimate_huber_op import estimate_huber_op, create_operators, Operator
from estimate_l2 import estimate_l2

# def run_gradient_iterations(
#     x_init: np.ndarray,
#     n_iterations: int,
#     b: np.ndarray,
#     op: Operator,
#     gamma_grad: float,
#     lambda_param: float,
#     rhohub: float,
#     A: float = 1
# ) -> Tuple[np.ndarray, List[float]]:
#     """
#     Run gradient descent iterations.
    
#     Args:
#         x_init: Initial point
#         n_iterations: Number of iterations
#         b: Target signal
#         op: Operator containing direct and adjoint operations
#         gamma_grad: Step size
#         lambda_param: Lambda parameter
#         rhohub: Huber parameter
#         A: Identity operator (default=1)
        
#     Returns:
#         Tuple containing:
#         - Final solution
#         - List of criterion values at each iteration
#     """
#     zn = x_init.copy()
#     crits = []
    
#     for _ in range(n_iterations):
#         # Calculate L2 criterion and gradient
#         crit1, grad1 = estimate_l2(zn.reshape(-1, 1), b, A)
        
#         # Calculate Huber criterion and gradient
#         crit2, grad2 = estimate_huber_op(zn, np.zeros_like(zn), rhohub, op)
        
#         # Store total criterion
#         crits.append(float(crit1 + lambda_param * crit2))
        
#         # Update step
#         zn = zn - gamma_grad * (grad1.flatten() + lambda_param * grad2)
        
#     return zn, crits

# # Load data
# n = 200
# mat_data = loadmat('../data/signal1D_200_v2.mat')
# x0 = mat_data['x0'][0] * 10
# b = x0 + 0.7 * np.random.randn(*x0.shape)

# # Create operators
# op, _, _ = create_operators(n)

# # Calculate norm of difference matrix
# D = op.direct(np.eye(n))
# nD = np.linalg.norm(D, ord=2) ** 2

# # Algorithm parameters
# A = 1
# lambda_param = 0.7
# rhohub = 0.002

# # Calculate gradient descent parameters
# alpha_grad = 1
# rho_grad = 1
# beta_grad = 1/(1/rhohub * nD * lambda_param)
# gamma_grad = 2/(rho_grad + 1/alpha_grad + 1/beta_grad)

# # Define color for gradient descent plot
# hgrad = np.array([76, 0, 153])/255

# # Run gradient descent for different iteration counts
# plt.figure(figsize=(15, 5))
# iterations = [10, 100, 20000]
# final_crits = None

# for i, n_iter in enumerate(iterations, 1):
#     print(f'Running gradient descent for {n_iter} iterations...')
    
#     # Run gradient descent
#     yn_grad, crits = run_gradient_iterations(
#         b, n_iter, b, op, gamma_grad, lambda_param, rhohub
#     )
    
#     # Store criterions from longest run for convergence plot
#     if n_iter == max(iterations):
#         final_crits = crits
    
#     # Plot results
#     plt.subplot(1, 3, i)
#     plt.plot(b, color='0.8', label='Noisy')
#     plt.plot(x0, 'k', label='Original')
#     plt.plot(yn_grad, color=hgrad, linewidth=1.2, label='EA')
#     plt.grid(True)
#     plt.title(f'{n_iter} iterations')
#     if i == 1:
#         plt.legend(fontsize=18)

# plt.tight_layout()

# # Plot convergence rate
# plt.figure()
# tau = gamma_grad
# rG = max(abs(1 - tau * rho_grad), 
#          abs(1 - tau * (1/beta_grad + 1/alpha_grad)))

# # Mask value where condition isn't met (equivalent to MATLAB's NaN assignment)
# if tau > 2 * alpha_grad * beta_grad / (beta_grad + alpha_grad):
#     rG = np.nan

# # Plot experimental criterion values
# plt.semilogy(final_crits, color=hgrad, linewidth=2, label='Experimental EA')

# # Plot theoretical rate
# it = np.arange(len(final_crits))
# plt.semilogy(rG**it * final_crits[0], color=hgrad, linewidth=4, 
#             linestyle='--', label='Theoretical EA')

# plt.xlim([0, 6000])
# plt.ylim([1.79329634085052e-12, 2])
# plt.legend(fontsize=16)
# plt.grid(True)
# plt.title('Convergence Rate')
# plt.xlabel('Iterations')
# plt.ylabel('Criterion Value')

# plt.show()

def ensure_plots_directory() -> str:
    """Create and return path to plots directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = f"plots_ea_{timestamp}"
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def run_gradient_iterations(
    x_init: np.ndarray,
    n_iterations: int,
    b: np.ndarray,
    op: Operator,
    gamma_grad: float,
    lambda_param: float,
    rhohub: float,
    A: float = 1
) -> Tuple[np.ndarray, List[float]]:
    """
    Run gradient descent iterations.
    
    Args:
        x_init: Initial point
        n_iterations: Number of iterations
        b: Target signal
        op: Operator containing direct and adjoint operations
        gamma_grad: Step size
        lambda_param: Lambda parameter
        rhohub: Huber parameter
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
        
        # Update step
        zn = zn - gamma_grad * (grad1.flatten() + lambda_param * grad2)
        
    return zn, crits

def plot_original_vs_noisy(x0: np.ndarray, b: np.ndarray, plots_dir: str) -> None:
    """
    Create and save a plot comparing original and noisy signals.
    
    Args:
        x0: Original signal
        b: Noisy signal
        plots_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(x0, color='black', linewidth=2, label='Señal original')
    plt.plot(b, color='0.8', alpha=0.8, linewidth=1.2, label='Señal con ruido')
    plt.grid(True)
    plt.title('Señal original y señal con ruido')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(fontsize=12)
    
    # Save plot
    plt.savefig(os.path.join(plots_dir, 'original_vs_noisy.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create plots directory
    plots_dir = ensure_plots_directory()
    
    # Load data
    n = 200
    mat_data = loadmat('../data/signal1D_200_v2.mat')
    x0 = mat_data['x0'][0] * 10
    b = x0 + 0.7 * np.random.randn(*x0.shape)

    # Create the original vs noisy signal plot
    plot_original_vs_noisy(x0, b, plots_dir)

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

    # Define color for gradient descent plot
    hgrad = np.array([76, 0, 153])/255

    # Run gradient descent for different iteration counts
    plt.figure(figsize=(15, 5))
    iterations = [10, 100, 20000]
    final_crits = None

    for i, n_iter in enumerate(iterations, 1):
        print(f'Running gradient descent for {n_iter} iterations...')
        
        # Run gradient descent
        yn_grad, crits = run_gradient_iterations(
            b, n_iter, b, op, gamma_grad, lambda_param, rhohub
        )
        
        # Store criterions from longest run for convergence plot
        if n_iter == max(iterations):
            final_crits = crits
        
        # Plot results
        plt.subplot(1, 3, i)
        plt.plot(b, color='0.8', label='Noisy')
        plt.plot(x0, 'k', label='Original')
        plt.plot(yn_grad, color=hgrad, linewidth=1.2, label='EA')
        plt.grid(True)
        plt.title(f'{n_iter} iterations')
        if i == 1:
            plt.legend(fontsize=18)

    plt.tight_layout()
    
    # Save iteration comparison plot
    plt.savefig(os.path.join(plots_dir, 'ea_iterations_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # Plot convergence rate
    plt.figure()
    tau = gamma_grad
    rG = max(abs(1 - tau * rho_grad), 
             abs(1 - tau * (1/beta_grad + 1/alpha_grad)))

    # Mask value where condition isn't met (equivalent to MATLAB's NaN assignment)
    if tau > 2 * alpha_grad * beta_grad / (beta_grad + alpha_grad):
        rG = np.nan

    # Plot experimental criterion values
    plt.semilogy(final_crits, color=hgrad, linewidth=2, label='Experimental EA')

    # Plot theoretical rate
    it = np.arange(len(final_crits))
    plt.semilogy(rG**it * final_crits[0], color=hgrad, linewidth=4, 
                linestyle='--', label='Theoretical EA')

    plt.xlim([0, 6000])
    plt.ylim([1.79329634085052e-12, 2])
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.title('Convergence Rate')
    plt.xlabel('Iterations')
    plt.ylabel('Criterion Value')

    # Save convergence rate plot
    plt.savefig(os.path.join(plots_dir, 'ea_convergence_rate.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots have been saved in the '{plots_dir}' directory.")

if __name__ == "__main__":
    main()