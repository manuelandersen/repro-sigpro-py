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

from function_huber_l2 import function_huber_l2
from grad_huberl2 import grad_huber_l2
from mask_param import mask_param
from amr2D_vect import amr2D_vect
from iamr2D_vect import iamr2D_vect
from prox_huberl2 import prox_huber_l2

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

        nb_it = 5

        
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

        if beta<=4*alpha:
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


        ######################
        # 5- Douglas-rachford
        ######################
        print('5- Douglas-rachford')


plt.figure(figsize=(15, 12))

# Original image
plt.subplot(221)
plt.imshow(x.reshape(int(sx), int(sx)), cmap='bone')
plt.axis('off')
plt.title('Original Image')

# Noisy/Degraded image
plt.subplot(222)
degraded = (A.T @ b).reshape(int(sx), int(sx))
plt.imshow(degraded, cmap='bone')
plt.axis('off')
plt.title('Degraded Image')

# Plot all three restorations in a row
plt.subplot(223)
plt.imshow(xn.reshape(int(sx), int(sx)), cmap='bone')  # FBS result
plt.axis('off')
plt.title('FBS Restored')

plt.subplot(224)
restored_drs = yn.reshape(int(sx), int(sx))  # DRS result
plt.imshow(restored_drs, cmap='bone')
plt.axis('off')
plt.title('DRS Restored')

plt.tight_layout()
plt.show()

# Convergence rates comparison
plt.figure(figsize=(10, 6))

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

# Plot experimental rates
plt.semilogy(critfbinf, color=hfb3, linewidth=1.2, label='FBS Experimental')
plt.semilogy(critprinf, color=hpr, linewidth=1.2, label='PRS Experimental')
plt.semilogy(critdrinf, color=hdr, linewidth=1.2, label='DRS Experimental')

# Plot theoretical rates
it = np.arange(max(len(critfbinf), len(critprinf), len(critdrinf)))
plt.semilogy(rFB3**it * rate0_fb, color=hfb3, linewidth=2, 
             linestyle='--', label='FBS Theoretical')
plt.semilogy(r_PR**it * rate0_pr, color=hpr, linewidth=2, 
             linestyle='--', label='PRS Theoretical')
plt.semilogy(r_DR**it * rate0_dr, color=hdr, linewidth=2, 
             linestyle='--', label='DRS Theoretical')

# Customize plot
plt.grid(True)
plt.xlim([0, 200])
plt.ylim([min(min(critfbinf), min(critprinf), min(critdrinf))*0.1, 
         max(max(critfbinf), max(critprinf), max(critdrinf))*10])
plt.legend(fontsize=12)
plt.xlabel('Iterations')
plt.ylabel('Criterion Value (log scale)')
plt.title('Convergence Rate Comparison')

plt.tight_layout()
plt.show()