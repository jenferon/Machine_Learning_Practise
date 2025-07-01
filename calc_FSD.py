import numpy as np
import sys
sys.path.append('/home/ppxjf3/ST4Diffusion_edited/')
import scattering
import torch
"""
load up simulated and created dataset
"""
original_image_tensor = torch.zeros([64,64,64])
for ii in range(0,64):
    original_image = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/data/Tb_coeval_z12_Tvir_eta_4.4_131.341_{}.npy'.format(ii+1)))
    original_image_tensor[ii,:,:] = torch.from_numpy(original_image)


#diffusion modeled array into pytorch tensor
diff_array = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/outputs/train-60xscale_0.1_test_test01_ema.npy'))
print(diff_array.shape)
generated_image_tensor = torch.zeros([64,64,64])
for ii in range(0,diff_array.shape[1]):
    generated_image_tensor[ii,:,:] =torch.from_numpy(diff_array[0,ii,0,:,:])

"""
Calculate the scattering coefficients

J=5
L=4
j2>j1

calculate the 0th, 1st and 2nd scattering coefficients between the two datasets
correct if dimension == 46
"""
M = N = 64; J = 5; L = 4
st_calc = scattering.Scattering2d(M, N, J, L) 

mean_x = st_calc.scattering_coef(original_image_tensor)['S2']
std_x  = st_calc.scattering_cov(original_image_tensor)['C11_iso']

mean_y = st_calc.scattering_coef(generated_image_tensor)['S2']
std_y  = st_calc.scattering_cov(generated_image_tensor)['C11_iso']

#log10 output - should be all greater than 0 
#This will help weight all coefficients equally
mean_x = torch.log10(mean_x)
mean_y = torch.log10(mean_y)
std_x = torch.log10(std_x)
std_y = torch.log10(std_y)
"""
Use the scattering distances to calculate FSD
"""

FSD = torch.abs(mean_x - mean_y)**2 + torch.trace(std_x + std_y - 2*torch.sqrt(std_x*std_y))
print(FSD)