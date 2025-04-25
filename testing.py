import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import functools
import matplotlib.pyplot as plt 
from torchvision.utils import save_image
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
IMG_size_learning = 256
IMG_size_output = 128

# explicit function to normalize array
def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix
  
def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The sigma in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t, device=DEVICE)

def match_tensor_shape(source, target):
    """
    Pads or crops `source` tensor so that it matches the spatial dimensions of `target`.
    
    Parameters:
    - source: Tensor to be adjusted (B, C, H1, W1)
    - target: Reference tensor to match shape to (B, C, H2, W2)
    
    Returns:
    - Adjusted source tensor with shape (B, C, H2, W2)
    """
    src_h, src_w = source.shape[-2:]
    tgt_h, tgt_w = target.shape[-2:]

    # Compute difference
    diff_h = tgt_h - src_h
    diff_w = tgt_w - src_w

    # Padding if target is larger
    if diff_h > 0 or diff_w > 0:
        pad_top = diff_h // 2
        pad_bottom = diff_h - pad_top
        pad_left = diff_w // 2
        pad_right = diff_w - pad_left
        source = F.pad(source, [pad_left, pad_right, pad_top, pad_bottom])
    # Cropping if source is larger
    elif diff_h < 0 or diff_w < 0:
        crop_top = -diff_h // 2
        crop_left = -diff_w // 2
        crop_bottom = crop_top + tgt_h
        crop_right = crop_left + tgt_w
        source = source[..., crop_top:crop_bottom, crop_left:crop_right]
    
    return source

#classes for use within U-net
class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]

#U-Net
class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[16, 32, 64, 128, IMG_size_learning], embed_dim=IMG_size_learning):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(8, num_channels=channels[1])
    
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(16, num_channels=channels[2])
    
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
    
    self.conv5 = nn.Conv2d(channels[3], channels[4], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[4])
    self.gnorm5 = nn.GroupNorm(32, num_channels=channels[4])

    # Decoding layers where the resolution increases
    self.tconv5 = nn.ConvTranspose2d(channels[4], channels[3], 3, stride=2, bias=False)
    self.dense6 = Dense(embed_dim, channels[3])
    self.tgnorm5 = nn.GroupNorm(32, num_channels=channels[3])
    
    self.tconv4 = nn.ConvTranspose2d(channels[3]*2, channels[2], 3, stride=2, bias=False)
    self.dense7 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(16, num_channels=channels[2])
    
    self.tconv3 = nn.ConvTranspose2d(channels[2]*2, channels[1], 3, stride=2, bias=False)    
    self.dense8 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(8, num_channels=channels[1])
    
    self.tconv2 = nn.ConvTranspose2d(channels[1]*2, channels[0], 3, stride=2, bias=False)    
    self.dense9 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(4, num_channels=channels[0])
    
    self.tconv1 = nn.ConvTranspose2d(channels[0]*2, 1, 3, stride=1, padding=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))    # Encoder
    h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
    h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
    h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
    h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))
    h5 = self.act(self.gnorm5(self.conv5(h4) + self.dense5(embed)))

    # Decoder
    h = self.act(self.tgnorm5(self.tconv5(h5) + self.dense6(embed)))
    h = match_tensor_shape(h, h4)
    h = self.act(self.tgnorm4(self.tconv4(torch.cat([h, h4], dim=1)) + self.dense7(embed)))
    h = match_tensor_shape(h, h3)
    h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense8(embed)))
    h = match_tensor_shape(h, h2)
    h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense9(embed)))
    h = match_tensor_shape(h, h1)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    h = F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=False)
    return h

class CustomImageDataset(Dataset):
    def __init__(self, base, data_name, transform=None, target_transform=None):
       self.transform = transform
       self.target_transform = target_transform
       self.image_loc = os.path.join(base, data_name)
       # Open the file in binary read mode ('rb')
       self.fid = open(self.image_loc, 'rb')
       self.dtype_size = 4  # Assuming float32 data type

    def __len__(self):
       # Get file size and calculate number of images
       file_size = os.path.getsize(self.image_loc)
       num_images = file_size // (600 * 600 * self.dtype_size)
       return num_images # Corrected length calculation

    def __getitem__(self, idx):
        # Calculate the starting position for the slice
        offset = idx * 600 * 600 * self.dtype_size

        # Seek to the correct position in the file
        self.fid.seek(offset)

        # Read only the required slice
        data = self.fid.read(600 * 600 * self.dtype_size)

        # Unpack the data for the slice into a NumPy array
        image = struct.unpack('f' * (600 * 600), data)
        image = np.array(image).reshape(600, 600)
        #image = np.expand_dims(image, axis=0)
        image = normalize_2d(image[:IMG_size_learning,:IMG_size_learning])

        # Convert to PyTorch tensor and add channel dimension
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        label = 'z=12.0'
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

    def __del__(self):
        # Close the file when the dataset object is deleted
        self.fid.close()
        
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


sigma =  1.5
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

batch_size = 30

dataset = CustomImageDataset('/home/ppxjf3/diffusion_GAN/','deltaTb_z12.000_N600_L200.0.dat')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

torch.manual_seed(123) # seed data set so always get the same output
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(DEVICE)

optimizer = Adam(score_model.parameters(), lr=1e-5)

tqdm_epoch = tqdm.trange(10)
for epoch in tqdm_epoch:
  avg_loss = 0.
  num_items = 0
  for x, y in data_loader:
    x = x.to(DEVICE)
    loss = loss_fn(score_model, x, marginal_prob_std_fn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
  # Print the averaged training loss so far.
  tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))


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
save_image(img1, 'img1.png')

img5 = samples[4] 
save_image(img5, 'img5.png')

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
plt.savefig('power_spec_validation.pdf', dpi=330, bbox_inches='tight')

#figure of sample and input side by side same colourbar and same scale
fig, ax = plt.subplots(1,2)
pmc = ax[0].imshow(dataset.__getitem__(0).cpu().detach().numpy()[:IMG_size_output,:IMG_size_output])
fig.colorbar(pmc, ax=ax[0])
pmc = ax[1].imshow(samples[idx].cpu().detach().numpy())
fig.colorbar(pmc, ax=ax[1])
plt.tight_layout()
plt.savefig('side_by_side_comparison.pdf', dpi=330, bbox_inches='tight')