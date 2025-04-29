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
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
n_gpus = torch.cuda.device_count()
print(n_gpus)
IMG_size_learning = 256
IMG_size_output = 128

#agrparse variables
parser = argparse.ArgumentParser()
parser.add_argument('-e','--nepoch',type=int, required=True)
parser.add_argument('-b','--batch', type=int, required=True)
args = parser.parse_args()

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
        x = x.squeeze(1)  # from [batch, 1, embed_dim] → [batch, embed_dim]
        x = self.dense(x)  # [batch, out_channels]
        return x[:, :, None, None]  # [batch, out_channels, 1, 1]



#U-Net with multiple residual blocks

class ScoreNet(nn.Module):
    def __init__(self, marginal_prob_std, base_channels=16, channel_mults=[1, 2, 4, 8, 16], embed_dim=32): #try changing embedded dim when it's working - defines the noise structure chat gpt suggests 128 to start
        """
        Args:
            marginal_prob_std: Function giving std of perturbation kernel at time t
            channels: List of output channels per encoder layer
            embed_dim: Dimensionality of time embedding
        """
        super().__init__()
        self.channels = [base_channels * m for m in channel_mults]
        self.embed_dim = embed_dim
        self.marginal_prob_std = marginal_prob_std

        # Embed time t using a Fourier projection followed by a linear layer
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Swish activation
        self.act = lambda x: x * torch.sigmoid(x)

        # Build encoder and decoder layers
        self.encoder_layers = self._build_encoder()
        self.decoder_layers = self._build_decoder()

        # Final layer to produce single-channel output
        self.final_conv = nn.ConvTranspose2d(self.channels[0]*2, 1, 3, stride=1, padding=1)

    def _build_encoder(self):
        layers = nn.ModuleList()
        in_channels = 1
        num_groups = [4,8,16,32,32]
        for ii, out_channels in enumerate(self.channels):
            if ii == 0:
                conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, bias=False, padding=1)
            else:
                conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, bias=False, padding=1)
            dense = Dense(self.embed_dim, out_channels)
            gnorm = nn.GroupNorm(num_groups=num_groups[ii], num_channels=out_channels) #figure out what to put for num_groups and what it does
            layers.append(nn.ModuleDict({'conv': conv, 'dense': dense, 'gnorm': gnorm}))
            in_channels = out_channels
        return layers

    def _build_decoder(self):
        layers = nn.ModuleList()
        reversed_channels = list(reversed(self.channels))
        in_channels = self.channels[-1] 
        
        for ii, out_channels in enumerate(reversed_channels[1:]):
            if ii == 0:
                tconv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, bias=False)
            else:
                tconv = nn.ConvTranspose2d(in_channels*2, out_channels, 3, stride=2, bias=False)
            dense = Dense(self.embed_dim, out_channels)
            gnorm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
            layers.append(nn.ModuleDict({'tconv': tconv, 'dense': dense, 'gnorm': gnorm}))
            in_channels = out_channels  # update for next layer
        return layers

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        # Encoder path
        skips = []
        h = x
        for layer in self.encoder_layers:
            h = self.act(layer['gnorm'](layer['conv'](h) + layer['dense'](embed)))
            skips.append(h)
        
        # Decoder path
        #rint("h is of size: {}".format(h.shape))
        h = skips.pop()  # bottleneck
        for layer in self.decoder_layers:
            
            h = self.act(layer['gnorm'](layer['tconv'](h) + layer['dense'](embed))) #issue
            skip = skips.pop()
            h = match_tensor_shape(h, skip)
            h = torch.cat([h, skip], dim=1)
        h = self.final_conv(h)

        # Normalize and resize
        h = h / self.marginal_prob_std(t)[:,None,None,None]
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
  #rint(perturbed_x.shape)
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss


sigma =  1.5
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

batch_size = args.batch

#split the data for training and validation
dataset = CustomImageDataset('/home/ppxjf3/diffusion_GAN/','deltaTb_z12.000_N600_L200.0.dat')

train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

#seed so always get the same
torch.manual_seed(123) 
score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
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

torch.save(score_model.state_dict(),'/home/ppxjf3/diffusion_GAN/diffusion_model.pth')
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
save_image(img1, 'img1_class.png')

img5 = samples[4] 
save_image(img5, 'img5_class.png')

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
plt.savefig('power_spec_validation_class.pdf', dpi=330, bbox_inches='tight')

#figure of sample and input side by side same colourbar and same scale
fig, ax = plt.subplots(1,2)
img, lab = dataset.__getitem__(idx)
img = img.cpu().detach().numpy()
pmc = ax[0].imshow(img[0,:IMG_size_output,:IMG_size_output])
fig.colorbar(pmc, ax=ax[0])
pmc = ax[1].imshow(samples[idx].cpu().detach().numpy()[0,:,:])
fig.colorbar(pmc, ax=ax[1])
plt.tight_layout()
plt.savefig('side_by_side_comparison.pdf', dpi=330, bbox_inches='tight')