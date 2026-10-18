import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SAE(nn.Module):
  def __init__(self, act_size: int = 768, exp_size: int = 8, k: int = 50):
    self.act_size = act_size # size of Hubert activations (768 for Hubert, 1024 for large Hubert)
    self.exp_size = exp_size # amount SAE expands the activations
    self.k = k # top k parameters taken from the expansion

    self.pre_bias = nn.Parameter(torch.zeroes(act_size))
    self.encoder = nn.Linear(act_size, act_size*exp_size)
    self.decoder = nn.Linear(act_size, act_size*exp_size)
    
    with torch.no_grad():
      self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim = 0)
      
    def encode(self, x: torch.Tensor) -> torch.Tensor:
      # encode the features by expanding the activation
      features = F.relu(self.encoder(x-self.pre_bias))

      # use only the top k features
      topk_values, topk_indices = torch.topk(features, k = self.k, dim = -1)
      features = torch.zeros_like(features)
      features.scatter_(dim=1, index = topk_indices, src = topk_values)
      return features
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
      reconstruction = self.decoder(x) + self.pre_bias
      return reconstruction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
      features = self.encode(x); reconstruction = self.decode(features)
      return features, reconstruction

    @torch.no_grad()
    def normalize_decoder(self):
      self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim = 0)

# still have to code the loss and train function
