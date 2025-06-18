import numpy as np
import matplotlib.pyplot as plt
import scattering
from PIL import Image
import torch
from torch import randint
from torchmetrics.image.kid import KernelInceptionDistance

generated_image = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/outputs/train-40xscale_0.1_test_test01_ema.npy'))
print(generated_image.shape)
generated_image_tensor = torch.zeros([64,1,64,64])
for ii in range(0,generated_image.shape[1]):
    generated_image_tensor[ii,:,:,:] =torch.from_numpy( generated_image[0,ii,0,:,:])

"""for ii in range(0,generated_image.shape[1]):
    im = Image.fromarray( generated_image[0,ii,0,:,:])
    im = im.convert('RGB')
    im.save("/home/ppxjf3/ST4Diffusion_edited/outputs/images_for_FID/Tb_generated_{}z.png".format(ii))"""

"""fig, ax = plt.subplots(1,1)
im = ax.imshow(original_image)
plt.colorbar(im, ax=ax)
plt.savefig("/home/ppxjf3/ST4Diffusion/outputs/original_image.png", dpi=330)"""
#for ii in range(0,original_image.shape[1]):
ii = 0
fig, ax = plt.subplots(1,1)
im = ax.imshow(generated_image[0,ii,0,:,:])
plt.colorbar(im, ax=ax)
plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/output_image{}.png".format(ii), dpi=330)
plt.close()

original_image_tensor = torch.zeros([64,1,64,64])
for ii in range(0,64):
    original_image = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/data/Tb_coeval_z12_Tvir_eta_4.4_131.341_{}x.npy'.format(ii)))
    original_image_tensor[ii,:,:,:] = torch.from_numpy(original_image)
    """im = Image.fromarray( generated_image[0,ii,0,:,:])
    im = im.convert('RGB')
    im.save("/home/ppxjf3/ST4Diffusion_edited/data/for_FID/Tb_original_{}x.png".format(ii))
    original_image = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/data/Tb_coeval_z12_Tvir_eta_4.4_131.341_{}y.npy'.format(ii)))
    im = Image.fromarray( generated_image[0,ii,0,:,:])
    im = im.convert('RGB')
    im.save("/home/ppxjf3/ST4Diffusion_edited/data/for_FID/Tb_original_{}y.png".format(ii))
    original_image = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/data/Tb_coeval_z12_Tvir_eta_4.4_131.341_{}z.npy'.format(ii)))
    im = Image.fromarray( generated_image[0,ii,0,:,:])
    im = im.convert('RGB')
    im.save("/home/ppxjf3/ST4Diffusion_edited/data/for_FID/Tb_original_{}z.png".format(ii))"""
#print(original_image_tensor.shape)

fig, ax = plt.subplots(1,1)
im = ax.imshow(original_image)
plt.colorbar(im, ax=ax)
plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/input_image{}.png".format(ii), dpi=330)
plt.close()

# 1-to-1 generation means that we generate one field matching the statistics of one given field. 
# if the input is a set of fields, then the 1-to-1 mode will synthesise them independently

# load image
image_target = original_image
# synthesize
image_syn = generated_image[0,ii,0,:,:]

# show
hist_range=(-2, 2)
hist_bins=50
plt.figure(figsize=(9,3), dpi=200)
plt.subplot(131) 
plt.imshow(image_target)
plt.xticks([]); plt.yticks([]); plt.title('original field')
plt.subplot(132)
plt.imshow(image_syn)
plt.xticks([]); plt.yticks([]); plt.title('modeled field')
plt.subplot(133); 
plt.hist(image_target.flatten(), hist_bins, histtype='step', label='target')
plt.hist(image_syn.flatten(), hist_bins, histtype='step', label='synthesized')
plt.yscale('log'); plt.legend(loc='lower center'); plt.title('histogram')
plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/histogram.png", dpi=330)
plt.close()

#comparitive power spectra
"""
Compare power spectra of input vs output
"""
import tools21cm as t2c
from astropy.cosmology import FlatLambdaCDM, LambdaCDM

import astropy.units as u
cosmo = FlatLambdaCDM(H0=71 * u.km / u.s / u.Mpc, Om0=0.27)

def return_power_spectra(data, length, kbins=12, binning='log'):
    box_dims = length
    V = length*length

    p, k = t2c.power_spectrum_1d(data[:,:],  box_dims=box_dims, kbins=kbins, binning = binning, return_n_modes=False)
    return (p*V*k**3)/(2*np.pi**2), k
  
#find variance of power spectra in samples
power_spec_sample = np.empty([generated_image.shape[1], 12])
k_sample = np.empty([12])
for ii in range(0,generated_image.shape[1]):
  power_spec_sample[ii,:], k_sample = return_power_spectra(generated_image[0,ii,0,:,:], 128)

power_spec_train, k_train = return_power_spectra(original_image, 128)
plt.plot(k_sample, np.mean(power_spec_sample, axis=0), label='sample')
plt.plot(k_train, power_spec_train, label='train')
plt.ylabel(r'$\Delta_{\rm 2D} ^2(k)/\rm mK^2$')
plt.xlabel(r'$k/(\rm Mpc/h)^{-1}$')
plt.yscale('log')
plt.xscale('log')
plt.legend(frameon=False, fontsize = 14)
plt.tight_layout()
plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/power_spec_comparison.png", dpi=330)
plt.close()

#KID

kid = KernelInceptionDistance(subset_size=64, degree=1)
kid.update(original_image_tensor.to(torch.uint8), real=True) #real images
kid.update(generated_image_tensor.to(torch.uint8), real=False) #generated images
print(kid.compute())