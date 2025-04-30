import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os
import pandas as pd
from skimage import io, transform
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import functools
import matplotlib.pyplot as plt 
from torchvision.utils import save_image
from PIL import Image
import argparse
import tools21cm as t2c
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
import astropy.units as u
cosmo = FlatLambdaCDM(H0=71 * u.km / u.s / u.Mpc, Om0=0.27)

from dataset import CustomImageDataset
from models import ScoreNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
n_gpus = torch.cuda.device_count()
print(n_gpus)
IMG_size_learning = 256
IMG_size_output = 128
BASE = "/home/ppxjf3/diffusion_GAN/"

#agrparse variables
parser = argparse.ArgumentParser()
parser.add_argument('-e','--nepoch',type=int, default=10)
parser.add_argument('-b','--batch', type=int, default=32)
parser.add_argument('-nres','--nres', default=1, type=int)
parser.add_argument('-b_ch','--base_channel', default =16, type=int)
args = parser.parse_args()

def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The sigma in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t, device=DEVICE)

def marginal_prob_std(t, sigma):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:
    t: A vector of time steps.
    sigma: The sigma in our SDE.

  Returns:
    The standard deviation.
  """
  t = torch.tensor(t, device=DEVICE)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss
sigma =  25
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

batch_size = args.batch

#split the data for training and validation
dataset = CustomImageDataset(BASE,"deltaTb_z12.000_N600_L200.0.dat", IMG_size_learning)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

#seed so always get the same
torch.manual_seed(123) 
score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn, base_channels=args.base_channel, num_blocks=args.nres)
score_model = torch.nn.DataParallel(score_model)
score_model = score_model.to(DEVICE)

optimizer = Adam(score_model.parameters(), lr=1e-5)


tqdm_epoch = tqdm.trange(args.nepoch)
avg_train_loss = np.empty([args.nepoch])
avg_validation_loss = np.empty([args.nepoch])

for idx, epoch in enumerate(tqdm_epoch):
  avg_loss = 0.
  num_items = 0
  for x, y in train_loader:
    x = x.to(DEVICE)
    loss = loss_fn(score_model, x, marginal_prob_std_fn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
  # Print the averaged training loss so far.
  avg_train_loss[idx] = avg_loss / num_items
  tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_train_loss[idx]))
  
  score_model.eval()  # Set the model to evaluation mode
  avg_val_loss = 0.
  num_val_items = 0
  with torch.no_grad():  # Disable gradient calculation for validation
    for x, y in val_loader:
      x = x.to(DEVICE)
      loss = loss_fn(score_model, x, marginal_prob_std_fn)
      avg_val_loss += loss.item() * x.shape[0]
      num_val_items += x.shape[0]
      
  avg_validation_loss[idx] = avg_val_loss / num_val_items
  print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss[idx]:.4f}, Validation Loss = {avg_validation_loss[idx]:.4f}')

#plot training and validation loss - use to check for overfitting
plt.figure()
plt.plot(np.arange(1, args.nepoch + 1),avg_validation_loss, label='validation')
plt.plot(np.arange(1, args.nepoch + 1),avg_train_loss, label='training')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(frameon=False, fontsize = 14)
plt.tight_layout()
plt.savefig('loss_plot.pdf', dpi=330, bbox_inches='tight')
plt.close()

score_model.eval()
test_loss = 0.
num_test_items = 0
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        loss = loss_fn(score_model, x, marginal_prob_std_fn)
        test_loss += loss.item() * x.shape[0]
        num_test_items += x.shape[0]

test_loss /= num_test_items
print(f"Test Loss: {test_loss:.4f}")

torch.save(score_model.state_dict(),BASE+"diffusion_model.pth")

""" IMAGE GENERATION  

Need to first define:
(1) the Pytorch model
(2) the standard deviation of the pertubation kernel
"""

#Define the ODE sampler (double click to expand or collapse)

from scipy import integrate
from tqdm.notebook import tqdm

def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=100, 
                           device='cuda', 
                           eps=1e-3):
  """Generate samples from score-based models with the Euler-Maruyama solver and applies
     a 2nd order Heun’s method

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """
  t = torch.ones(batch_size, device=device)
  x = torch.randn(batch_size, 1, IMG_size_output, IMG_size_output, device=device)* marginal_prob_std(t)[:, None, None, None]
  
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  
  with torch.no_grad():
    for i  in tqdm(range(num_steps - 1)):      
      batch_time_step = time_steps[i].expand(batch_size)
      g = diffusion_coeff(batch_time_step)
      #rint(batch_time_step.shape)
      score = score_model(x, batch_time_step)
      x_euler = x + (g**2)[:, None, None, None] * score * step_size
      x_euler += torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
      
      # Corrector (Heun)
      t_next = time_steps[i + 1].expand(batch_size)
      x = x + 0.5 * (g**2)[:, None, None, None] * (score + score_model(x_euler, t_next)) * step_size

  return x

sample_batch_size = 5
sampler = Euler_Maruyama_sampler #what was used in kerras and camebridge paper 

## Generate samples using the specified sampler.
samples = sampler(score_model, 
                  marginal_prob_std_fn,
                  diffusion_coeff_fn, 
                  sample_batch_size, 
                  device=DEVICE)

print(samples.shape)

img1 = samples[0] 
save_image(img1, BASE+"img1_class.png")

img5 = samples[4] 
save_image(img5, BASE+"img5_class.png")

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
power_spec_sample = np.empty([len(samples), 12])
k_sample = np.empty([12])
for idx in range(0,len(samples)):
  power_spec_sample[idx,:], k_sample = return_power_spectra(samples[idx].cpu().detach().numpy(), 200.0*(IMG_size_output/600))


#power spectrum for the training data
power_spec_train = np.empty([len(samples), 12])
k_train = np.empty([12])
for idx in range(0,len(samples)):
  img, lab = dataset.__getitem__(idx)
  power_spec_train[idx,:], k_train = return_power_spectra(img.cpu().detach().numpy(), 200.0*(IMG_size_learning/600))
  
plt.plot(k_sample, np.mean(power_spec_sample, axis=0), label='sample')
plt.plot(k_train, np.mean(power_spec_train, axis=0), label='train')
plt.ylabel(r'$\Delta_{\rm 2D} ^2(k)/\rm mK^2$')
plt.xlabel(r'$k/(\rm Mpc/h)^{-1}$')
plt.yscale('log')
plt.xscale('log')
plt.legend(frameon=False, fontsize = 14)
plt.tight_layout()
plt.savefig(BASE+"power_spec_validation_class.pdf", dpi=330, bbox_inches='tight')
plt.close()
#figure of sample and input side by side same colourbar and same scale
fig, ax = plt.subplots(1,2)
img, lab = dataset.__getitem__(idx)
img = img.cpu().detach().numpy()
pmc = ax[0].imshow(img[0,:IMG_size_output,:IMG_size_output])
fig.colorbar(pmc, ax=ax[0])
pmc = ax[1].imshow(samples[idx].cpu().detach().numpy()[0,:,:])
fig.colorbar(pmc, ax=ax[1])
plt.tight_layout()
plt.savefig(BASE+"side_by_side_comparison.pdf", dpi=330, bbox_inches='tight')
plt.close()