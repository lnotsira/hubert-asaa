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

  def compute_loss(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
      l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
      l1_norm = acts_topk.float().abs().sum(-1).mean()
      l1_loss = self.cfg["l1_coeff"] * l1_norm
      l0_norm = (acts_topk > 0).float().sum(-1).mean()
      aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
      loss = l2_loss + l1_loss + aux_loss
      num_dead_features = (
          self.num_batches_not_active > self.cfg["n_batches_to_dead"]
      ).sum()
      sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
      output = {
          "sae_out": sae_out,
          "feature_acts": acts_topk,
          "num_dead_features": num_dead_features,
          "loss": loss,
          "l1_loss": l1_loss,
          "l2_loss": l2_loss,
          "l0_norm": l0_norm,
          "l1_norm": l1_norm,
          "aux_loss": aux_loss,
      }
      return output

"""
  def get_auxiliary_loss(self, x, x_reconstruct, acts):
      dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
      if dead_features.sum() > 0:
          residual = x.float() - x_reconstruct.float()
          acts_topk_aux = torch.topk(
              acts[:, dead_features],
              min(self.cfg["top_k_aux"], dead_features.sum()),
              dim=-1,
          )
          acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
              -1, acts_topk_aux.indices, acts_topk_aux.values
          )
          x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
          l2_loss_aux = (
              self.cfg["aux_penalty"]
              * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
          )
          return l2_loss_aux
      else:
          return torch.tensor(0, dtype=x.dtype, device=x.device)
"""

  def train_sae(sae, activations, num_epochs = 100, batch_size = 256, lr = 3e-4):
      optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      sae = sae.to(device)
      activations = activations.to(device)
      
      for epoch in range(num_epochs):
          indices = torch.randperm(len(activations))
          shuffled = activations[indices]
          
          epoch_losses = []
          for i in range(0, len(activations), batch_size):
              batch = shuffled[i : i+batch_size]
              
              sparse_features, reconstruction = sae(batch)
              loss_dict = compute_loss(batch, sparse_features, reconstruction)
              
              optimizer.zero_grad()
              loss_dict['total_loss'].backward()
              optimizer.step()
              
              if i % 100 == 0:
                  sae.normalize_decoder()
              
              epoch_losses.append(loss_dict['recon_loss'].item())
          
          if (epoch + 1) % 10 == 0:
              avg_loss = np.mean(epoch_losses)
              print(f"Epoch {epoch+1}, Loss: {avg_loss}")
      
      return sae
