import sys
import os
sys.path.append(os.path.abspath('../utilities'))

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict
from scipy.io import loadmat
from datetime import datetime

from estimate_huber_op import estimate_huber_op, create_operators, Operator, create_difference_matrix
from estimate_l2 import estimate_l2
from prox_huberl2_matrix import prox_huber_l2_matrix
from function_huber_l2 import function_huber_l2
from prox_l2diff import prox_l2diff

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
    plt.plot(b, color='green', alpha=0.8, linewidth=1.2, label='Señal con ruido')
    plt.grid(True)
    plt.title('Señal original y señal con ruido')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(fontsize=12)
    
    # Save plot
    plt.savefig(os.path.join(plots_dir, 'original_vs_noisy.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def ensure_plots_directory() -> str:
    """Create and return path to plots directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = f"plot_algos_{timestamp}"
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


# Create plots directory
plots_dir = ensure_plots_directory()

# Load data
n = 200
mat_data = loadmat('../data/signal1D_200_v2.mat')
x0 = mat_data['x0'][0] * 10
b = x0 + 0.7 * np.random.randn(*x0.shape)

# Create the original vs noisy signal plot
plot_original_vs_noisy(x0, b, plots_dir)

t = np.ones((n, 1))
D = np.zeros((n, n)) 
D = create_difference_matrix(n)
nD = np.linalg.norm(D, ord=2) ** 2
D1 = D[::2]  # odd-indexed rows
nD1 = np.linalg.norm(D1, ord=2) ** 2
D2 = D[1::2]  # even-indexed rows
nD2 = np.linalg.norm(D2, ord=2) ** 2

A = 1
op, op1, op2 = create_operators(n)
D1_diag = np.diag(D1 @ D1.T)
D2_diag = np.diag(D2 @ D2.T)


lambda_val = 0.7
rhohub = 0.002 #0.0001

nb_it = 10000


######################################
# PR infty
######################################

print("PR infty")
alpha_pr = 1/(1 + 1/rhohub * nD2 * lambda_val)
rho_pr = 1
beta_pr = 1/(1/rhohub * nD1 * lambda_val)
gamma_pr = np.sqrt(alpha_pr/rho_pr)

_, initgrad = estimate_huber_op(b, np.zeros(len(b)//2), rhohub, op1)
xn = b + gamma_pr * lambda_val * initgrad

crit_prbar = []
with tqdm(total=nb_it, desc='PR infty Progress') as pbar:
        for it in range(nb_it):
            yn = xn - op1.adjoint(1/D1_diag * (op1.direct(xn) - prox_huber_l2_matrix(op1.direct(xn), rhohub, gamma_pr * lambda_val * D1_diag)))
            tmp = (2*yn - xn + gamma_pr*b)/(1 + gamma_pr)
            zn = tmp - op2.adjoint(1/D2_diag * (op2.direct(tmp) - prox_huber_l2_matrix(op2.direct(tmp), rhohub, gamma_pr * lambda_val * D2_diag/(1 + gamma_pr))))
            xn = 2*zn - (2*yn - xn)
            crit_prbar.append(0.5 * np.sum((A*yn - b)**2) + lambda_val * function_huber_l2(op.direct(yn), rhohub))
            pbar.update(1)

xbarpr = xn
rate0pr = np.linalg.norm(b + gamma_pr * lambda_val * initgrad - xbarpr)
ybar = yn
rate0 = np.linalg.norm(b - ybar)


######################################
# DR infty
######################################
print("DR infty")

alpha_dr = 1/(1 + 1/rhohub * nD2 * lambda_val)
rho_dr = 1
beta_dr = 1/(1/rhohub * nD1 * lambda_val)

if beta_dr < 4 * alpha_dr/(1 + np.sqrt(1 - alpha_dr * rho_dr)**2):
    gamma_dr = np.sqrt(alpha_dr/rho_dr)
else:
    gamma_dr = np.sqrt(beta_dr/rho_dr)

# Get initial point
_, initgrad = estimate_huber_op(b, np.zeros(len(b)//2), rhohub, op1)

xn = b + gamma_dr * lambda_val * initgrad

crit_drbar = []
with tqdm(total=nb_it, desc='PR infty Progress') as pbar:
        for it in range(nb_it):
            yn = xn - op1.adjoint(1/D1_diag * (op1.direct(xn) - prox_huber_l2_matrix(op1.direct(xn), rhohub, gamma_dr * lambda_val * D1_diag)))
            tmp = (2*yn - xn + gamma_dr*b)/(1 + gamma_dr)
            zn = tmp - op2.adjoint(1/D2_diag * (op2.direct(tmp) - prox_huber_l2_matrix(op2.direct(tmp), rhohub, gamma_dr * lambda_val * D2_diag/(1 + gamma_dr))))
            xn = xn + zn - yn
            pbar.update(1)
            crit_drbar.append(0.5 * np.sum((A*yn - b)**2) + lambda_val * function_huber_l2(op.direct(yn), rhohub))
            
xbardr = xn
rate0dr = np.linalg.norm(xn - xbardr)


######################################
# Gradiente
######################################
print("Gradiente")

nn = 1
rate_grad = []
crit_grad = []
for nb_it in [10,100,10000]:
    with tqdm(total=nb_it, desc=f'Gradiente nb_it {nb_it}') as pbar:
            zn = b.T
            alpha_grad = 1
            rho_grad = 1
            beta_grad = 1 / (1 / rhohub * nD * lambda_val)
            gamma_grad = 2 / (rho_grad + 1 / alpha_grad + 1 / beta_grad)
            for it in range(nb_it):
                rate_grad.append(np.linalg.norm(zn - ybar))
                crit1, grad1 = estimate_l2(zn, b, A)
                crit2, grad2 = estimate_huber_op(zn, np.zeros_like(zn), rhohub, op)
                #crit_grad(it) = crit1 + lambda*crit2 ;
                crit_grad.append(crit1 + lambda_val * crit2)
                pbar.update(1)

yn_grad = zn

###################################################
# Forward-backward (2-):prox_hubL1 et grad l2+hubL2
###################################################
print("Forward-backward (2-):prox_hubL1 et grad l2+hubL2")

# TODO: que significa esto en frances(?)
# FB : x0 = b et ||xn - ybar|| et ||x0 - ybar||         

xn =b.T
alpha_fb2 = 1 / (1 + 1 / rhohub * nD2 * lambda_val) 
rho_fb2 = 1
beta_fb2 = 1 / (1 / rhohub * nD1 * lambda_val)  
gamma_fb2 = 2 / (rho_fb2 + 1 / alpha_fb2)

rate_fb2 = []
crit_fb2 = []
with tqdm(total=nb_it, desc='Forward-backward (2-)') as pbar:
        for it in range(nb_it):
            rate_fb2.append(np.linalg.norm(xn - ybar))
            crit1, grad1 = estimate_huber_op(xn, np.zeros(len(xn)//2), rhohub, op2)
            crit2 = function_huber_l2(op1.direct(xn), rhohub)
            crit3, grad3 = estimate_l2(xn, b, A)
            #crit_fb2(it) = crit3 + lambda*crit2 + lambda* crit1;   
            crit_fb2.append(crit3 + lambda_val * crit2 + lambda_val * crit1)
            tmp = xn - gamma_fb2* (lambda_val * grad1 + grad3)
            #xn = tmp - op1.adjoint(1./D1.*(op1.direct(tmp) - prox_huberl2_matrix(op1.direct(tmp), rhohub, gamma_fb2*lambda*D1)));
            xn = tmp - op1.adjoint(1/D1_diag * (op1.direct(tmp) - prox_huber_l2_matrix(op1.direct(tmp), rhohub, gamma_fb2 * lambda_val * D1_diag)))
            pbar.update(1)

yn_fb2 = xn

###################################################
# Forward-backward (3a-): prox_l2 et grad hub
###################################################
print("Forward-backward (3a-): prox_l2 et grad hub")

xn = b.T
alpha_fb3a = 1
rho_fb3a = 1
beta_fb3a = 1 / (1 / rhohub * nD * lambda_val)  
gamma_fb3a = 1.99 * beta_fb3a

Minv = 1 / (1 + gamma_fb3a)

rate_fb3a = []
crit_fb3a = []
with tqdm(total=nb_it, desc='Forward-backward (3a-)') as pbar:
        for it in range(nb_it):
            rate_fb3a.append(np.linalg.norm(xn - ybar))
            crit2, grad2 = estimate_huber_op(xn, np.zeros(n), rhohub, op)
            crit_fb3a.append(0.5 * np.sum((A * xn.ravel() - b) ** 2) + lambda_val * crit2)
            xn = prox_l2diff(xn - gamma_fb3a * lambda_val * grad2, b, A, gamma_fb3a, Minv)
            pbar.update(1)
    
yn_fb3a = xn


###################################################
# Forward-backward (3b-):prox_l2+hubL2 et grad hubL1
###################################################
print("Forward-backward (3b-):prox_l2+hubL2 et grad hubL1")

# TODO: que significa esto en frances(?)
# FB : x0 = b et ||xn - ybar|| et ||x0 - ybar||           

alpha_fb3b = 1 / (1 + 1 / rhohub * nD2 * lambda_val)
rho_fb3b = 1
beta_fb3b = 1 / (1 / rhohub * nD1 * lambda_val)
gamma_fb3b = 2 * beta_fb3b
xn =b.T

rate_fb3b = []
crit_fb3b = []
with tqdm(total=nb_it, desc='Forward-backward (3b-)') as pbar:
        for it in range(nb_it):
            rate_fb3b.append(np.linalg.norm(xn - ybar))
            crit1, grad1 = estimate_huber_op(xn, np.zeros(len(xn)//2), rhohub, op1)
            crit2 = function_huber_l2(op2.direct(xn), rhohub)
            crit3, _ = estimate_l2(xn, b, A)
            #crit_fb3b(it) = crit3 + lambda*crit2 + lambda* crit1;   
            crit_fb3b.append(crit3 + lambda_val * crit2 + lambda_val * crit1)
            tmp = (xn - gamma_fb3b * (lambda_val * grad1) + gamma_fb3b * b.T) / (1 + gamma_fb3b)
            xn = tmp - op2.adjoint(1/D2_diag * (op2.direct(tmp) - prox_huber_l2_matrix(op2.direct(tmp), rhohub, gamma_fb3b * lambda_val * D2_diag/(1 + gamma_fb3b))))
            pbar.update(1)

yn_fb3b = xn


###################################################
# Peaceman-Rachford
###################################################
print("Peaceman-Rachford")

alpha_pr = 1 / (1 + 1 / rhohub * nD2 * lambda_val)
rho_pr = 1
beta_pr = 1 / (1 / rhohub * nD1 * lambda_val) 
gamma_pr = np.sqrt(alpha_pr / rho_pr)
_, initgrad = estimate_huber_op(b, np.zeros(len(b)//2), rhohub, op1)
xn = b.T + gamma_pr * lambda_val * initgrad
rate0pr = np.linalg.norm(xn - xbarpr.ravel())

rate_pr = []
crit_pr = []
with tqdm(total=nb_it, desc='Peaceman-Rachford') as pbar:
        for it in range(nb_it):
            #rate_pr(it) = norm(xn(:) - xbarpr(:),'fro');  %% ATTENTION LE xbar n'est probablement pas le bon
            rate_pr.append(np.linalg.norm(xn.ravel() - xbarpr.ravel()))
            yn = xn - op1.adjoint(1/D1_diag * (op1.direct(xn) - prox_huber_l2_matrix(op1.direct(xn), rhohub, gamma_pr * lambda_val * D1_diag)))
            tmp = (2 * yn - xn + gamma_pr * b.T) / (1 + gamma_pr)
            #zn = tmp - op2.adjoint(1./D2.*(op2.direct(tmp) - prox_huberl2_matrix(op2.direct(tmp), rhohub, gamma_pr*lambda.*D2./(1+gamma_pr))));
            zn = tmp - op2.adjoint(1/D2_diag * (op2.direct(tmp) - prox_huber_l2_matrix(op2.direct(tmp), rhohub, gamma_pr * lambda_val * D2_diag/(1 + gamma_pr))))
            xn = 2 * zn - (2 * yn - xn)
            crit_pr.append(0.5 * np.sum((A * yn.ravel() - b) ** 2) + lambda_val * function_huber_l2(op.direct(yn), rhohub))
            pbar.update(1)

yn_pr = yn

###################################################
# Douglas-Rachford
###################################################
print("Douglas-Rachford")

alpha_dr = 1 / (1 + 1 / rhohub * nD2 * lambda_val)
rho_dr = 1
beta_dr = 1 / (1 / rhohub * nD1 * lambda_val)

if beta_dr < 4 * alpha_dr/(1 + np.sqrt(1 - alpha_dr * rho_dr)**2):
    gamma_dr = np.sqrt(alpha_dr/rho_dr)
else:
    gamma_dr = np.sqrt(beta_dr/rho_dr)

_, initgrad = estimate_huber_op(b, np.zeros(len(b)//2), rhohub, op1)

xn = b + gamma_dr * lambda_val * initgrad
rate0dr = np.linalg.norm(xn - xbardr.ravel())

rate_dr = []
crit_dr = []
with tqdm(total=nb_it, desc='Douglas-Rachford') as pbar:
        for it in range(nb_it):
            rate_dr.append(np.linalg.norm(xn.ravel() - xbardr.ravel()))
            yn = xn - op1.adjoint(1 / D1_diag * (op1.direct(xn) - prox_huber_l2_matrix(op1.direct(xn), rhohub, gamma_dr * lambda_val * D1_diag)))
            tmp = (2 * yn - xn + gamma_dr * b.T) / (1 + gamma_dr)
            zn = tmp - op2.adjoint(1/D2_diag * (op2.direct(tmp) - prox_huber_l2_matrix(op2.direct(tmp), rhohub, gamma_dr * lambda_val * D2_diag/(1 + gamma_dr))))
            xn = xn + zn-yn
            crit_dr.append(0.5 * np.sum((A*yn - b)**2) + lambda_val * function_huber_l2(op.direct(yn), rhohub))
            pbar.update(1)
yn_dr = yn



def plot_algorithm_results(results_data, plots_dir):
    """
    Plot algorithm results and convergence rates.
    """
    # Colors for different algorithms (matching MATLAB code)
    colors = {
        'grad': np.array([76, 0, 153])/255,    # EA/Gradient
        'fb2': np.array([153, 0, 153])/255,    # FBS2
        'fb3a': np.array([218, 152, 207])/255, # FBS3a
        'fb3b': np.array([255, 0, 127])/255,   # FBS3b
        'pr': np.array([100, 29, 29])/255,     # PRS
        'dr': np.array([192, 135, 70])/255     # DRS
    }

    # Plot results for different iteration counts
    plt.figure(figsize=(15, 5))
    iterations = [10000]
    
    for i, n_iter in enumerate(iterations, 1):
        plt.subplot(1, 3, i)
        
        # Plot noisy and original signals
        plt.plot(results_data['b'], color='0.8', label='Noisy')
        plt.plot(results_data['x0'], 'k', label='Original')
        
        # Plot results for each algorithm
        plt.plot(results_data['yn_grad'], color=colors['grad'], linewidth=1.2, label='EA')
        plt.plot(results_data['yn_fb3a'], color=colors['fb3a'], linestyle='--', linewidth=1.2, label='FBS')
        plt.plot(results_data['yn_fb2'], color=colors['fb2'], linewidth=1.2, label='FBS2')
        plt.plot(results_data['yn_fb3b'], color=colors['fb3b'], linestyle='--', linewidth=1.2, label='FBS3')
        plt.plot(results_data['yn_pr'], color=colors['pr'], linewidth=1.2, label='PRS')
        plt.plot(results_data['yn_dr'], color=colors['dr'], linewidth=1.2, label='DRS')
        
        plt.grid(True)
        plt.title(f'{n_iter} iterations')
        if i == 1:
            plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot convergence rates
    plt.figure(figsize=(10, 6))
    
    # Calculate theoretical rates
    tau = results_data['gamma_grad']
    rG = max(abs(1 - tau * results_data['rho_grad']),
            abs(1 - tau * (1/results_data['beta_grad'] + 1/results_data['alpha_grad'])))
    if tau > 2 * results_data['alpha_grad'] * results_data['beta_grad'] / (results_data['beta_grad'] + results_data['alpha_grad']):
        rG = np.nan

    # FB2
    tau = results_data['gamma_fb2']
    rFB2 = max(abs(1 - tau * results_data['rho_fb2']), abs(1 - tau/results_data['alpha_fb2']))

    # FB3a
    tau = results_data['gamma_fb3a']
    rFB3a = 1/(1 + tau * results_data['rho_fb3a'])

    # FB3b
    tau = results_data['gamma_fb3b']
    rFB3b = 1/(1 + tau * results_data['rho_fb3b'])

    # PR
    tau = results_data['gamma_pr']
    rPR = max((1 - tau * results_data['rho_pr'])/(1 + tau * results_data['rho_pr']),
              (tau/results_data['alpha_pr'] - 1)/(tau/results_data['alpha_pr'] + 1))

    # DR
    tau = results_data['gamma_dr']
    rDR = min(1/2 + 1/2 * max((1 - tau * results_data['rho_dr'])/(1 + tau * results_data['rho_dr']),
                              (tau/results_data['alpha_dr'] - 1)/(tau/results_data['alpha_dr'] + 1)),
              (results_data['beta_dr'] + tau**2 * results_data['rho_dr'])/
              (results_data['beta_dr'] + tau * results_data['beta_dr'] * results_data['rho_dr'] + tau**2 * results_data['rho_dr']))

    # Plot experimental rates
    plt.semilogy(results_data['rate_grad'], color=colors['grad'], linewidth=1.2, label='EA Exp')
    plt.semilogy(results_data['rate_fb3a'], color=colors['fb3a'], linewidth=1.2, label='FBS Exp')
    plt.semilogy(results_data['rate_fb2'], color=colors['fb2'], linewidth=1.2, label='FBS2 Exp')
    plt.semilogy(results_data['rate_fb3b'], color=colors['fb3b'], linewidth=1.2, label='FBS3 Exp')
    plt.semilogy(results_data['rate_pr'], color=colors['pr'], linewidth=1.2, label='PRS Exp')
    plt.semilogy(results_data['rate_dr'], color=colors['dr'], linewidth=1.2, label='DRS Exp')

    # Plot theoretical rates
    it = np.arange(len(results_data['rate_grad']))
    plt.semilogy(rG**it * results_data['rate0'], color=colors['grad'], linewidth=2, linestyle='-.', label='EA Theo')
    
    it = np.arange(len(results_data['rate_fb3a']))
    plt.semilogy(rFB3a**it * results_data['rate0'], color=colors['fb3a'], linewidth=2, linestyle='-.', label='FBS Theo')
    
    it = np.arange(len(results_data['rate_fb2']))
    plt.semilogy(rFB2**it * results_data['rate0'], color=colors['fb2'], linewidth=2, linestyle='-.', label='FBS2 Theo')
    
    it = np.arange(len(results_data['rate_fb3b']))
    plt.semilogy(rFB3b**it * results_data['rate0'], color=colors['fb3b'], linewidth=2, linestyle='-.', label='FBS3 Theo')
    
    it = np.arange(len(results_data['rate_pr']))
    plt.semilogy(rPR**it * results_data['rate0pr'], color=colors['pr'], linewidth=2, linestyle='-.', label='PRS Theo')
    
    it = np.arange(len(results_data['rate_dr']))
    plt.semilogy(rDR**it * results_data['rate0dr'], color=colors['dr'], linewidth=2, linestyle='-.', label='DRS Theo')

    plt.xlim([0, 6000])
    plt.ylim([1.79329634085052e-12, 2])
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.title('Convergence Rates')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/convergence_rates.png', dpi=300, bbox_inches='tight')
    plt.close()


# After running all algorithms, collect results
results_data = {
    # Input data
    'x0': x0,
    'b': b,
    
    # Algorithm results
    'yn_grad': yn_grad,
    'yn_fb2': yn_fb2,
    'yn_fb3a': yn_fb3a,
    'yn_fb3b': yn_fb3b,
    'yn_pr': yn_pr,
    'yn_dr': yn_dr,
    
    # Convergence rates
    'rate_grad': rate_grad,
    'rate_fb2': rate_fb2,
    'rate_fb3a': rate_fb3a,
    'rate_fb3b': rate_fb3b,
    'rate_pr': rate_pr,
    'rate_dr': rate_dr,
    
    # Initial rates
    'rate0': rate0,
    'rate0pr': rate0pr,
    'rate0dr': rate0dr,
    
    # Algorithm parameters
    'alpha_grad': alpha_grad,
    'rho_grad': rho_grad,
    'beta_grad': beta_grad,
    'gamma_grad': gamma_grad,
    
    'alpha_fb2': alpha_fb2,
    'rho_fb2': rho_fb2,
    'gamma_fb2': gamma_fb2,
    
    'alpha_fb3a': alpha_fb3a,
    'rho_fb3a': rho_fb3a,
    'gamma_fb3a': gamma_fb3a,
    
    'alpha_fb3b': alpha_fb3b,
    'rho_fb3b': rho_fb3b,
    'gamma_fb3b': gamma_fb3b,
    
    'alpha_pr': alpha_pr,
    'rho_pr': rho_pr,
    'gamma_pr': gamma_pr,
    
    'alpha_dr': alpha_dr,
    'rho_dr': rho_dr,
    'beta_dr': beta_dr,
    'gamma_dr': gamma_dr
}

# Generate plots
plot_algorithm_results(results_data, plots_dir)