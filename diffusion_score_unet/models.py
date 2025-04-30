import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import pandas as pd
import functools
from torch.nn.functional import silu
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#classes for use within U-net
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
  
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch, n_groups=1, stride=2, up=False, down=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.embed_ch = emb_ch
        
        self.up = up 
        self.down = down
        #first normalisation and conv
        self.norm1 = nn.GroupNorm(n_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        
        # Embed -> scale and shift
        self.emb_proj = nn.Linear(emb_ch, out_ch * 2)

        # Second normalization and conv
        self.norm2 = nn.GroupNorm(n_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # Optional skip connection conv if in/out channels differ
        self.skip = None
        if in_ch != out_ch or up or down:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        self.downsample = None
        self.upsample = None
        if down:
            self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        if up:
            self.upsample = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x, emb):
        
        #print(x.shape)
        h = self.conv1(silu(self.norm1(x)))
        
        # Inject embedding: scale and shift
        emb_out = self.emb_proj(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)
        #print(h.shape)

        # Residual connection
        skip = self.skip(x) if self.skip is not None else x
        h = h + skip
        #print(h.shape)

        if self.downsample is not None:
            h = self.downsample(h)
        elif self.upsample is not None:
            h = self.upsample(h)

        return h
    
class ScoreNet(nn.Module):
    def __init__(self, marginal_prob_std, base_channels=16, channel_mults=[1, 2, 4, 8, 16], num_blocks = 1, embed_dim=32): #try changing embedded dim when it's working - defines the noise structure chat gpt suggests 128 to start
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

        # Build encoder and decoder layers
        #self.encoder_layers = self._build_encoder()
        self.encoder_layers = nn.ModuleList()
        in_ch = 1
        #n_groups = [4,8,16,32,32]
        for i, out_ch in enumerate(self.channels):
            blocks = []
            for j in range(num_blocks):
                is_down = (j == num_blocks - 1)  # Only last block downsamples
                stride = 2 if is_down and i != 0 else 1
                blocks.append(UNetBlock(in_ch, out_ch, embed_dim, stride=stride, down=is_down))
                in_ch = out_ch  # Update for next block
            #print(in_ch)
            self.encoder_layers.append(nn.ModuleList(blocks))
            
        self.decoder_layers = nn.ModuleList()
        reversed_channels = list(reversed(self.channels))
        in_ch = reversed_channels[0]
        for i, out_ch in enumerate(reversed_channels[1:]):
            blocks = []
            for j in range(num_blocks):
                is_up = (j == 0)  # Only first block upsamples
                if j == 0:
                    if i == 0:
                        block_in_ch = in_ch  # no skip connection yet
                    else:
                        block_in_ch = in_ch * 2  # concat with skip
                else:
                    block_in_ch = out_ch  # internal block
                stride = 2 if is_up  else 1
                # First decoder block gets concatenated skip input
                blocks.append(UNetBlock(block_in_ch, out_ch, embed_dim, up=is_up, stride=stride, down=False))
            self.decoder_layers.append(nn.ModuleList(blocks))
            in_ch = out_ch

        # Final layer to produce single-channel output
        self.final_conv = nn.ConvTranspose2d(self.channels[0]*2, 1, 3, stride=1, padding=1)
        
    def forward(self, x, t):
        embed = silu(self.embed(t))
        skips = []
        h = x
        #print("enter encoder")
        for stage in self.encoder_layers:
            for block in stage:
                h = block(h, embed)
            skips.append(h)
            #print("embedding skip of shape {}".format(h.shape))

        h = skips.pop()  # bottleneck
        #print('bottleneck')
        #print("h of shape {}".format(h.shape))
        
        #print("enter decoder")

        # Decoder
        for stage in self.decoder_layers:
            for block in stage:
                #print(h.shape)
                h = block(h, embed)
            skip = skips.pop()
            #print("skip shape: {}".format(skip.shape))
            #print("h shape {}".format(h.shape))
            h = match_tensor_shape(h, skip)
            h = torch.cat([h, skip], dim=1)   
        #print(h.shape)        
        # Normalize and resize
        h = self.final_conv(h)
        h = h / self.marginal_prob_std(t)[:,None,None,None]
        h = F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=True)
        return h