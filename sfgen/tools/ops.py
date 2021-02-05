import torch

def duplicate_vector(vector, n, dim=1):
  """B X D vector to B X N X D vector"""
  batch_size = vector.shape[:dim]
  return torch.ones(*batch_size, n, *vector.shape[dim:], device=vector.device)*vector.unsqueeze(dim)
