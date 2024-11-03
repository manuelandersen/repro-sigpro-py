import sys
import os
sys.path.append(os.path.abspath('../utilities'))

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.color import rgb2gray
from skimage import io
import pywt
from PIL import Image
from numpy.linalg import solve
import datetime
import pathlib

from function_huber_l2 import function_huber_l2
from grad_huberl2 import grad_huber_l2
from mask_param import mask_param
from amr2D_vect import amr2D_vect
from iamr2D_vect import iamr2D_vect
from prox_huberl2 import prox_huber_l2


# Número de iteraciones, ojo que se demora mucho en correr
# probar primero con un numero bajo. Dps ir subiendo.
NB_IT = 500



def ensure_output_directory(base_dir='output'):
    """Create timestamped output directory for saving results."""
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create timestamped subdirectory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(output_dir)
    
    return output_dir

def save_plot(fig, filename, output_dir, dpi=300):
    """Save figure to specified output directory with given filename."""
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)



hgrad = np.array([0, 0, 204]) / 255
hfb2 = np.array([153, 0, 153]) / 255
hfb3 = np.array([255, 51, 153]) / 255
hpr = np.array([100, 29, 29]) / 255
hdr = np.array([192, 135, 70]) / 255
col = plt.cm.hsv(np.linspace(0, 1, 7))

img = Image.open('IMG_5885.jpg').convert('L')
x0 = np.array(img, dtype=np.float64)
x = x0[762:762+64*2:2, 923:923+64*3:3]

sx = np.sqrt(x.size)
mat_data = loadmat('filter_gaussian_7_5_N64.mat')
A = mat_data['A']
sx1, sy1 = A.shape
A = np.random.randn(int(np.sqrt(sx1))**2, sy1)

print("Calculando eigvalues")
ev = np.linalg.eigvals(A.T @ A)

A = A / np.sqrt(np.max(ev))
ev = np.linalg.eigvals(A.T @ A)

vpmin = np.min(ev)
vpmax = np.max(ev)

alpha = vpmax
rho = vpmin
sig = 10

#TODO: no se cual de las dos es la manera correcta
pywt.Modes.from_object('periodization')  # Equivalent to MATLAB's dwtmode('per')
#pywt.Modes.modes['per'] = pywt.Modes.MODE_PERIODIC

#TODO: averiguar que configura esto
prox = {'rm': 3}
prox['wav'] = 'sym3' 

N = int(np.log2(sx))
mask = mask_param(N, prox['rm'], prox['wav']) + 1e-16

#TODO: añadir todo lo que es grafico adentro de este loop, y poner los 3 posibles
# valores de lambda. Asi mas facil graficar de una.
for lambda_val in [1e-3]:
        rhohub =1
        beta = rhohub/lambda_val

        #TODO: arreglar el reshape de b]
        b = (A @ x.flatten()).reshape(-1, 1) + sig * np.random.randn(4900, 1)
        sb = np.sqrt(np.size(b))

        op = {}
        op['direct'] = lambda x: mask * amr2D_vect(x[:int(sx*sx)].reshape(int(sx), int(sx)), prox['rm'], prox['wav'])
        op['adjoint'] = lambda x: iamr2D_vect(x/mask, prox['rm'], prox['wav']).reshape(int(sx*sx), 1)
        x0algo = A.T @ b 

        nb_it = NB_IT

        
        ######################
        # Forward-backward 
        ######################
        print('Forward-backward')

        xn = x0algo

        gamma_fb3 = 1.9 * beta

        n = int(sx*sx)  
        IplusAAinv = np.linalg.inv(np.eye(n, n) + gamma_fb3 * A.T @ A)

        critfbinf = []
        with tqdm(total=nb_it, desc='Forward-backward Progress') as pbar:
                for it in range(1, nb_it + 1):

                        if len(xn.shape) == 1:
                                xn = xn.reshape(-1, 1)
                        
                        xn_reshaped = xn.reshape(int(sx*sx), 1) 

                        crit2 = function_huber_l2(op['direct'](xn), rhohub)
                        crit1 = 0.5 * np.linalg.norm(A @ xn - b, 'fro')**2

                        grad1 = op['adjoint'](grad_huber_l2(op['direct'](xn), rhohub))

                        update = xn - gamma_fb3 * lambda_val * grad1 + gamma_fb3 * A.T @ b
                        xn = (IplusAAinv @ update).reshape(-1, 1) 
                        critfbinf.append(crit1 + lambda_val * crit2)
                        pbar.update(1)
        
        ybar = xn.copy()
        rate0_fb = np.linalg.norm(x0algo.flatten() - ybar.flatten())



        ######################
        # Peaceman-Rachford
        ######################
        print('Peaceman-Rachford')

        gamma_pr = np.sqrt(alpha/rho)
        n = int(sx*sx)  
        #TODO: pq este va sin la inversa ????
        IplusAAinv = np.eye(n, n) + gamma_fb3 * A.T @ A
        xn0 = x0algo + gamma_pr * A.T @ (A @ x0algo - b)
        xn = xn0

        critprinf = []
        with tqdm(total=nb_it, desc='Peaceman-Rachford Progress') as pbar:
                for it in range(1, nb_it + 1):
                       yn = solve(IplusAAinv, xn + gamma_pr * A.T @ b)
                       zn = op['adjoint'](prox_huber_l2(op['direct'](2 * yn - xn), rhohub, gamma_pr * lambda_val * mask**2))
                       xn = 2*zn-(2*yn-xn)
                       crit2 = function_huber_l2(op['direct'](yn), rhohub)
                       critprinf.append(0.5 * np.linalg.norm(A @ yn - b, 'fro')**2 + lambda_val * crit2)
                       pbar.update(1)
        
        xbarpr = xn
        rate0_pr = np.linalg.norm(xn0.flatten() - xbarpr.flatten())


        ######################
        # Douglas-Rachford
        ######################
        print('Douglas-Rachford')

        if beta <= 4 * alpha:
                gamma_dr = np.sqrt(alpha/rho)
        else:
               gamma_dr = np.sqrt(beta/rho)

        n = int(sx*sx)  
        IplusAAinv = np.eye(n,n) + gamma_dr * A.T @ A
        xn0 = x0algo + gamma_dr * A.T @ (A @ x0algo - b)
        xn = xn0

        critdrinf = []
        with tqdm(total=nb_it, desc='Douglas-Rachford Progress') as pbar:
                for it in range(1, nb_it + 1):
                       yn = np.linalg.solve(IplusAAinv, xn + gamma_dr * A.T @ b)
                       zn = op['adjoint'](prox_huber_l2(op['direct'](2*yn - xn), rhohub, gamma_dr * lambda_val * mask**2))
                       xn = xn + zn-yn
                       crit2 = function_huber_l2(op['direct'](yn), rhohub)
                       critdrinf.append(0.5 * np.linalg.norm(A @ yn - b, 'fro')**2 + lambda_val * crit2)
                       pbar.update(1)

        xbardr = xn
        rate0_dr = np.linalg.norm(xn0.flatten() - xbardr.flatten())


        ######################
        # 3-forward-backward
        ######################
        print('3-forward-backward')

        gamma_fb3 = 1.9*beta
        n = int(sx*sx)  
        IplusAAinv = np.linalg.inv(np.eye(n, n) + gamma_fb3 * A.T @ A)
        xn = x0algo

        crit_fb3 = []
        rate_fb3 = np.zeros(nb_it)
        with tqdm(total=nb_it, desc='3-forward-backward') as pbar:
                for it in range(1, nb_it + 1):
                       
                       if len(xn.shape) == 1:
                              xn = xn.reshape(-1, 1)
                       xn_reshaped = xn.reshape(int(sx*sx), 1) 

                       rate_fb3[it-1] = np.linalg.norm(xn.ravel() - ybar.ravel())
                       crit2 = function_huber_l2(op['direct'](xn), rhohub)
                       crit1 = 0.5 * np.linalg.norm(A @ xn - b, 'fro')**2
                       grad1 = op['adjoint'](grad_huber_l2(op['direct'](xn), rhohub))
                       #xn = IplusAAinv*(xn - gamma_fb3*lambda*grad1 +gamma_fb3 *A'*b);
                       update = xn - gamma_fb3 * lambda_val * grad1 + gamma_fb3 * A.T @ b
                       xn = (IplusAAinv @ update).reshape(-1, 1) 
                       crit_fb3.append(crit1 + lambda_val* crit2)
                       pbar.update(1)

        yn_fb3 = xn


        ######################
        # 4- Peaceman-Rachford
        ######################
        print('4- Peaceman-Rachford')

        gamma_pr = np.sqrt(alpha/rho)
        n = int(sx*sx)  
        IplusAAinv = np.linalg.inv(np.eye(n, n) + gamma_pr * A.T @ A)
        #xn0 = x0algo+gamma_pr*A'*(A*x0algo-b)
        xn0 = x0algo + gamma_pr * A.T @ (A @ x0algo - b)
        xn = xn0
        crit_pr = []
        rate_pr = np.zeros(nb_it)
        with tqdm(total=nb_it, desc='4- Peaceman-Rachford') as pbar:
                for it in range(1, nb_it + 1):
                        rate_pr[it-1] = np.linalg.norm(xn.ravel() - xbarpr.ravel())
                        update = xn + gamma_pr * A.T @ b
                        yn = (IplusAAinv @ update).reshape(-1, 1) 
                        #yn = IplusAAinv*(xn +gamma_pr*A'*b);
                        zn = op['adjoint'](prox_huber_l2(op['direct'](2*yn - xn), rhohub, gamma_pr * lambda_val * mask**2))
                        xn = 2 * zn - (2 * yn - xn)
                        crit2 = function_huber_l2(op['direct'](yn),rhohub)
                        crit_pr.append(0.5 * np.linalg.norm(A @ yn-b,'fro')**2 + lambda_val * crit2)  
                        pbar.update(1)

        yn_pr = yn

        ######################
        # 5- Douglas-rachford
        ######################
        print('5- Douglas-rachford')

        if beta <= 4 * alpha:
            gamma_dr = np.sqrt(alpha/rho)
        else:
            gamma_dr = np.sqrt(beta/rho)

        n = int(sx*sx)  
        IplusAAinv = np.linalg.inv(np.eye(n, n) + gamma_dr * A.T @ A)
        xn0 = x0algo + gamma_dr * A.T @ (A @ x0algo - b)
        xn = xn0

        crit_dr = []
        rate_dr = np.zeros(nb_it)
        with tqdm(total=nb_it, desc='5- Douglas-rachford') as pbar:
                for it in range(1, nb_it + 1):
                        rate_pr[it-1] = np.linalg.norm(xn.ravel() - xbardr.ravel())
                        update = xn + gamma_pr * A.T @ b
                        yn = (IplusAAinv @ update).reshape(-1, 1) 
                        #zn = op.adjoint(prox_huberl2(op.direct(2*yn-xn), rhohub, gamma_dr*lambda.*mask.^2));
                        zn = op['adjoint'](prox_huber_l2(op['direct'](2*yn-xn), rhohub, gamma_dr * lambda_val * mask**2))
                        xn = xn + zn-yn
                        crit2 = function_huber_l2(op['direct'](yn),rhohub)
                        crit_dr.append(0.5 * np.linalg.norm(A @ yn-b,'fro')**2 + lambda_val * crit2)
                        pbar.update(1)

        yn_dr = yn

# Calculate theoretical rates
# FBS theoretical rate
rho_fb3 = rho
tau_fb = gamma_fb3
rFB3 = 1/(1 + tau_fb * rho_fb3)
if tau_fb > 2 * beta:
    rFB3 = np.nan

# PRS theoretical rate
tau_pr = gamma_pr
r_PR = max((1 - tau_pr * rho)/(1 + tau_pr * rho),
           (tau_pr/alpha - 1)/(tau_pr/alpha + 1))

# DRS theoretical rate
tau_dr = gamma_dr
r_DR = min(1/2 + 1/2 * max((1 - tau_dr * rho)/(1 + tau_dr * rho),
                           (tau_dr/alpha - 1)/(tau_dr/alpha + 1)),
           (beta + tau_dr**2 * rho)/(beta + tau_dr * beta * rho + tau_dr**2 * rho))

# Calculate theoretical rates
# FBS theoretical rate
rho_fb3 = rho
tau_fb = gamma_fb3
rFB3 = 1/(1 + tau_fb * rho_fb3)
if tau_fb > 2 * beta:
    rFB3 = np.nan

# PRS theoretical rate
tau_pr = gamma_pr
r_PR = max((1 - tau_pr * rho)/(1 + tau_pr * rho),
           (tau_pr/alpha - 1)/(tau_pr/alpha + 1))

# DRS theoretical rate
tau_dr = gamma_dr
r_DR = min(1/2 + 1/2 * max((1 - tau_dr * rho)/(1 + tau_dr * rho),
                           (tau_dr/alpha - 1)/(tau_dr/alpha + 1)),
           (beta + tau_dr**2 * rho)/(beta + tau_dr * beta * rho + tau_dr**2 * rho))

# Create output directory
output_dir = ensure_output_directory()

# Plot results
fig1 = plt.figure(figsize=(15, 12))

# Original image
plt.subplot(231)
plt.imshow(x.reshape(int(sx), int(sx)), cmap='bone')
plt.axis('off')
plt.title('Original Image')

# Noisy/Degraded image
plt.subplot(232)
degraded = (A.T @ b).reshape(int(sx), int(sx))
plt.imshow(degraded, cmap='bone')
plt.axis('off')
plt.title('Degraded Image')

# Plot all restorations
plt.subplot(233)
plt.imshow(yn_fb3.reshape(int(sx), int(sx)), cmap='bone')
plt.axis('off')
plt.title('FB3 Restored')

plt.subplot(234)
plt.imshow(yn_pr.reshape(int(sx), int(sx)), cmap='bone')
plt.axis('off')
plt.title('PR Restored')

plt.subplot(235)
plt.imshow(yn_dr.reshape(int(sx), int(sx)), cmap='bone')
plt.axis('off')
plt.title('DR Restored')

plt.subplot(236)
plt.imshow(xn.reshape(int(sx), int(sx)), cmap='bone')
plt.axis('off')
plt.title('FB Restored')

plt.tight_layout()
save_plot(fig1, 'restoration_comparison.png', output_dir)

# Convergence rates comparison
fig2 = plt.figure(figsize=(12, 8))

# Plot experimental rates
plt.semilogy(crit_fb3, color=hfb3, linewidth=1.2, label='FB3 Experimental')
plt.semilogy(critfbinf, color=hfb2, linewidth=1.2, label='FB Experimental')
plt.semilogy(crit_pr, color=hpr, linewidth=1.2, label='PR Experimental')
plt.semilogy(crit_dr, color=hdr, linewidth=1.2, label='DR Experimental')
plt.semilogy(critprinf, color=col[4], linewidth=1.2, label='PR2 Experimental')

# Plot theoretical rates
it = np.arange(nb_it)

# Original methods theoretical
plt.semilogy(rFB3**it * rate0_fb, color=hfb3, linewidth=2, 
        linestyle='--', label='FB3 Theoretical')
plt.semilogy(r_PR**it * rate0_pr, color=hpr, linewidth=2, 
        linestyle='--', label='PR Theoretical')
plt.semilogy(r_DR**it * rate0_dr, color=hdr, linewidth=2, 
        linestyle='--', label='DR Theoretical')

# Additional methods theoretical
plt.semilogy(rFB3**it * rate0_fb, color=hfb2, linewidth=2, 
        linestyle='--', label='FB Theoretical')
plt.semilogy(r_PR**it * rate0_pr, color=col[4], linewidth=2, 
        linestyle='--', label='PR2 Theoretical')

# Customize plot
plt.grid(True)
plt.xlim([0, nb_it])
min_val = min([min(c) for c in [crit_fb3, critfbinf, crit_pr, crit_dr, critprinf] if len(c) > 0])
max_val = max([max(c) for c in [crit_fb3, critfbinf, crit_pr, crit_dr, critprinf] if len(c) > 0])
plt.ylim([min_val*0.1, max_val*10])

plt.legend(fontsize=10, ncol=2)
plt.xlabel('Iterations')
plt.ylabel('Criterion Value (log scale)')
plt.title('Convergence Rate Comparison')

plt.tight_layout()
save_plot(fig2, 'convergence_rates.png', output_dir)
