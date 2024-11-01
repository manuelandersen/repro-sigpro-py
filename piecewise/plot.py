import sys
import os
sys.path.append(os.path.abspath('../utilities'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from scipy.io import loadmat


from estimate_huber_op import estimate_huber_op, create_operators, Operator
from prox_huberl2_matrix import prox_huber_l2_matrix
from estimate_l2 import estimate_l2
from prox_l2diff import prox_l2diff
from function_huber_l2 import function_huber_l2

def setup_problem(n: int = 200) -> Tuple:
    """Setup common problem parameters and data."""
    # Load data
    mat_data = loadmat('../data/signal1D_200_v2.mat')
    x0 = mat_data['x0'][0] * 10
    b = x0 + 0.7 * np.random.randn(*x0.shape)
    
    # Create operators
    op, op1, op2 = create_operators(n)
    
    # Calculate matrices and norms
    D = op.direct(np.eye(n))
    D1 = D[::2]
    D2 = D[1::2]
    nD = np.linalg.norm(D, ord=2) ** 2
    nD1 = np.linalg.norm(D1, ord=2) ** 2
    nD2 = np.linalg.norm(D2, ord=2) ** 2
    
    # Calculate diagonal matrices
    D1_diag = np.diag(D1 @ D1.T)
    D2_diag = np.diag(D2 @ D2.T)
    
    return x0, b, op, op1, op2, D1_diag, D2_diag, nD, nD1, nD2

def run_drs(x_init: np.ndarray, params: Dict) -> Tuple[np.ndarray, List[float]]:
    """Run DRS iterations."""
    xn = x_init.copy()
    rates = []
    
    for _ in range(params['n_iter']):
        if params['xbar'] is not None:
            rates.append(np.linalg.norm(xn - params['xbar']))
            
        yn = xn - params['op1'].adjoint(1/params['D1_diag'] * (params['op1'].direct(xn) - 
             prox_huber_l2_matrix(params['op1'].direct(xn), params['rhohub'], 
                                 params['gamma'] * params['lambda_param'] * params['D1_diag'])))
        
        tmp = (2*yn - xn + params['gamma']*params['b'])/(1 + params['gamma'])
        
        zn = tmp - params['op2'].adjoint(1/params['D2_diag'] * (params['op2'].direct(tmp) - 
             prox_huber_l2_matrix(params['op2'].direct(tmp), params['rhohub'], 
                                 params['gamma'] * params['lambda_param'] * 
                                 params['D2_diag']/(1 + params['gamma']))))
        
        xn = xn + zn - yn
        
    return xn, rates

def run_prs(x_init: np.ndarray, params: Dict) -> Tuple[np.ndarray, List[float]]:
    """Run PRS iterations."""
    xn = x_init.copy()
    rates = []
    
    for _ in range(params['n_iter']):
        if params['xbar'] is not None:
            rates.append(np.linalg.norm(xn - params['xbar']))
            
        yn = xn - params['op1'].adjoint(1/params['D1_diag'] * (params['op1'].direct(xn) - 
             prox_huber_l2_matrix(params['op1'].direct(xn), params['rhohub'], 
                                 params['gamma'] * params['lambda_param'] * params['D1_diag'])))
        
        tmp = (2*yn - xn + params['gamma']*params['b'])/(1 + params['gamma'])
        
        zn = tmp - params['op2'].adjoint(1/params['D2_diag'] * (params['op2'].direct(tmp) - 
             prox_huber_l2_matrix(params['op2'].direct(tmp), params['rhohub'], 
                                 params['gamma'] * params['lambda_param'] * 
                                 params['D2_diag']/(1 + params['gamma']))))
        
        xn = 2*zn - (2*yn - xn)
        
    return xn, rates

def run_fbs2(x_init: np.ndarray, params: Dict) -> Tuple[np.ndarray, List[float]]:
    """Run FBS2 iterations."""
    xn = x_init.copy()
    rates = []
    
    for _ in range(params['n_iter']):
        if params['xbar'] is not None:
            rates.append(np.linalg.norm(xn - params['xbar']))
            
        _, grad1 = estimate_huber_op(xn, np.zeros(params['n']//2), params['rhohub'], params['op2'])
        _, grad3 = estimate_l2(xn.reshape(-1, 1), params['b'], params['A'])
        tmp = xn - params['gamma'] * (params['lambda_param'] * grad1 + grad3.flatten())
        xn = tmp - params['op1'].adjoint(1/params['D1_diag'] * (params['op1'].direct(tmp) - 
             prox_huber_l2_matrix(params['op1'].direct(tmp), params['rhohub'], 
                                 params['gamma'] * params['lambda_param'] * params['D1_diag'])))
        
    return xn, rates

def run_fbs3a(x_init: np.ndarray, params: Dict) -> Tuple[np.ndarray, List[float]]:
    """Run FBS3a iterations."""
    xn = x_init.copy()
    rates = []
    
    for _ in range(params['n_iter']):
        if params['xbar'] is not None:
            rates.append(np.linalg.norm(xn - params['xbar']))
        
        _, grad2 = estimate_huber_op(xn, np.zeros(params['n']), params['rhohub'], params['op'])
        xn = prox_l2diff(xn - params['gamma'] * params['lambda_param'] * grad2,
                        params['b'], params['A'], params['gamma'], params['Minv'])
        
    return xn, rates

def run_fbs3b(x_init: np.ndarray, params: Dict) -> Tuple[np.ndarray, List[float]]:
    """Run FBS3b iterations."""
    xn = x_init.copy()
    rates = []
    
    for _ in range(params['n_iter']):
        if params['xbar'] is not None:
            rates.append(np.linalg.norm(xn - params['xbar']))
            
        _, grad1 = estimate_huber_op(xn, np.zeros(params['n']//2), params['rhohub'], params['op1'])
        tmp = (xn - params['gamma'] * params['lambda_param'] * grad1 + 
               params['gamma'] * params['b'])/(1 + params['gamma'])
        xn = tmp - params['op2'].adjoint(1/params['D2_diag'] * (params['op2'].direct(tmp) - 
             prox_huber_l2_matrix(params['op2'].direct(tmp), params['rhohub'],
                                 params['gamma'] * params['lambda_param'] * 
                                 params['D2_diag']/(1 + params['gamma']))))
        
    return xn, rates

def run_ea(x_init: np.ndarray, params: Dict) -> Tuple[np.ndarray, List[float]]:
    """Run EA (gradient descent) iterations."""
    xn = x_init.copy()
    rates = []
    
    for _ in range(params['n_iter']):
        if params['xbar'] is not None:
            rates.append(np.linalg.norm(xn - params['xbar']))
            
        _, grad1 = estimate_l2(xn.reshape(-1, 1), params['b'], params['A'])
        _, grad2 = estimate_huber_op(xn, np.zeros_like(xn), params['rhohub'], params['op'])
        xn = xn - params['gamma'] * (grad1.flatten() + params['lambda_param'] * grad2)
        
    return xn, rates

def main():
    # Setup problem
    n = 200
    x0, b, op, op1, op2, D1_diag, D2_diag, nD, nD1, nD2 = setup_problem(n)
    
    # Common parameters
    A = 1
    lambda_param = 0.7
    rhohub = 0.0001 #la otra opcion es 0.0001
    
    # Algorithm-specific parameters
    # DRS parameters
    alpha_dr = 1/(1 + 1/rhohub * nD2 * lambda_param)
    rho_dr = 1
    beta_dr = 1/(1/rhohub * nD1 * lambda_param)
    gamma_dr = np.sqrt(alpha_dr/rho_dr)
    
    # PRS parameters
    alpha_pr = 1/(1 + 1/rhohub * nD2 * lambda_param)
    rho_pr = 1
    gamma_pr = np.sqrt(alpha_pr/rho_pr)
    
    # FBS parameters
    alpha_fb2 = 1/(1 + 1/rhohub * nD2 * lambda_param)
    rho_fb2 = 1
    beta_fb2 = 1/(1/rhohub * nD1 * lambda_param)
    gamma_fb2 = 2/(rho_fb2 + 1/alpha_fb2)
    
    # FBS3a parameters
    alpha_fb3a = 1
    rho_fb3a = 1
    beta_fb3a = 1/(1/rhohub * nD * lambda_param)
    gamma_fb3a = 1.99 * beta_fb3a
    Minv_fb3a = 1/(1 + gamma_fb3a)
    
    # FBS3b parameters
    alpha_fb3b = 1/(1 + 1/rhohub * nD2 * lambda_param)
    rho_fb3b = 1
    beta_fb3b = 1/(1/rhohub * nD1 * lambda_param)
    gamma_fb3b = 2 * beta_fb3b
    
    # EA parameters
    alpha_grad = 1
    rho_grad = 1
    beta_grad = 1/(1/rhohub * nD * lambda_param)
    gamma_grad = 2/(rho_grad + 1/alpha_grad + 1/beta_grad)
    
    # Setup colors
    colors = {
        'DRS': np.array([192, 135, 70])/255,
        'PRS': np.array([100, 29, 29])/255,
        'FBS2': np.array([153, 0, 153])/255,
        'FBS3a': np.array([218, 152, 207])/255,
        'FBS3b': np.array([255, 0, 127])/255,
        'EA': np.array([76, 0, 153])/255
    }
    
    # First run long iterations to get reference solutions
    nb_it_max = 500000
    base_params = {
        'n': n, 'A': A, 'lambda_param': lambda_param, 'rhohub': rhohub,
        'b': b, 'op': op, 'op1': op1, 'op2': op2,
        'D1_diag': D1_diag, 'D2_diag': D2_diag,
        'n_iter': nb_it_max, 'xbar': None
    }
    
    # Get reference solutions
    _, initgrad = estimate_huber_op(b, np.zeros(len(b)//2), rhohub, op1)
    
    params_dr = {**base_params, 'gamma': gamma_dr}
    x_dr, _ = run_drs(b + gamma_dr * lambda_param * initgrad, params_dr)
    
    params_pr = {**base_params, 'gamma': gamma_pr}
    x_pr, _ = run_prs(b + gamma_pr * lambda_param * initgrad, params_pr)
    
    params_fb2 = {**base_params, 'gamma': gamma_fb2}
    x_fb2, _ = run_fbs2(b, params_fb2)
    
    params_fb3a = {**base_params, 'gamma': gamma_fb3a, 'Minv': Minv_fb3a}
    x_fb3a, _ = run_fbs3a(b, params_fb3a)
    
    params_fb3b = {**base_params, 'gamma': gamma_fb3b}
    x_fb3b, _ = run_fbs3b(b, params_fb3b)
    
    params_ea = {**base_params, 'gamma': gamma_grad}
    x_ea, _ = run_ea(b, params_ea)
    
    # Plot results for different iteration counts
    iterations = [10, 100, 20000]
    plt.figure(figsize=(15, 5))
    
    for i, n_iter in enumerate(iterations, 1):
        plt.subplot(1, 3, i)
        plt.plot(b, color='0.8', label='Noisy')
        plt.plot(x0, 'k', label='Original')
        
        # Update parameters with new iteration count
        base_params['n_iter'] = n_iter
        
        # Run and plot each algorithm
        algorithms = {
            'DRS': (run_drs, x_dr, gamma_dr),
            'PRS': (run_prs, x_pr, gamma_pr),
            'FBS2': (run_fbs2, x_fb2, gamma_fb2),
            'FBS3a': (run_fbs3a, x_fb3a, gamma_fb3a),
            'FBS3b': (run_fbs3b, x_fb3b, gamma_fb3b),
            'EA': (run_ea, x_ea, gamma_grad)
        }
        
        for name, (func, xbar, gamma) in algorithms.items():
            params = {**base_params, 'gamma': gamma, 'xbar': xbar}
            if name == 'FBS3a':
                params['Minv'] = Minv_fb3a
            x_init = b if name.startswith('FBS') else b + gamma * lambda_param * initgrad
            xn, _ = func(x_init, params)
            plt.plot(xn, color=colors[name], linewidth=1.2, label=name)
        
        plt.grid(True)
        plt.title(f'{n_iter} iterations')
        if i == 1:
            plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Plot convergence rates
    plt.figure(figsize=(10, 6))
    n_iter = 20000
    base_params['n_iter'] = n_iter
    
    # Get all rates
    all_rates = {}
    for name, (func, xbar, gamma) in algorithms.items():
        params = {**base_params, 'gamma': gamma, 'xbar': xbar}
        if name == 'FBS3a':
            params['Minv'] = Minv_fb3a
        x_init = b if name.startswith('FBS') else b + gamma * lambda_param * initgrad
        _, rates = func(x_init, params)
        all_rates[name] = rates
    
    # Calculate theoretical rates
    it = np.arange(len(all_rates['DRS']))
    rate0 = np.linalg.norm(b - x_dr)  # Initial error for reference
    
    # DRS theoretical rate
    r_DR = min(1/2 + 1/2 * max((1 - gamma_dr * rho_dr)/(1 + gamma_dr * rho_dr),
                              (gamma_dr/alpha_dr - 1)/(gamma_dr/alpha_dr + 1)),
              (beta_dr + gamma_dr**2 * rho_dr)/(beta_dr + gamma_dr * beta_dr * rho_dr + gamma_dr**2 * rho_dr))
    
    # PRS theoretical rate
    tau = gamma_pr
    r_PR = max((1 - tau * rho_pr)/(1 + tau * rho_pr),
               (tau/alpha_pr - 1)/(tau/alpha_pr + 1))
    
    # FBS theoretical rates
    # FBS2
    tau = gamma_fb2
    r_FB2 = max(abs(1 - tau * rho_fb2), abs(1 - tau/alpha_fb2))
    
    # FBS3a
    tau = gamma_fb3a
    r_FB3a = 1/(1 + tau * rho_fb3a)
    
    # FBS3b
    tau = gamma_fb3b
    r_FB3b = 1/(1 + tau * rho_fb3b)
    
    # EA theoretical rate
    tau = gamma_grad
    r_EA = max(abs(1 - tau * rho_grad), 
              abs(1 - tau * (1/beta_grad + 1/alpha_grad)))
    if tau > 2 * alpha_grad * beta_grad / (beta_grad + alpha_grad):
        r_EA = np.nan
    
    # Plot experimental rates
    for name in algorithms:
        plt.semilogy(all_rates[name], color=colors[name], linewidth=1.2, 
                    label=f'{name} Experimental')
    
    # Plot theoretical rates
    theo_rates = {
        'DRS': r_DR,
        'PRS': r_PR,
        'FBS2': r_FB2,
        'FBS3a': r_FB3a,
        'FBS3b': r_FB3b,
        'EA': r_EA
    }
    
    for name, rate in theo_rates.items():
        if not np.isnan(rate):
            plt.semilogy(it, rate**it * rate0, color=colors[name], linewidth=2,
                        linestyle='-.', label=f'{name} Theoretical')
    
    plt.xlim([0, 6000])
    plt.ylim([1e-12, 2])
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.title('Algorithm Convergence Rates')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()

if __name__ == "__main__":
    main()