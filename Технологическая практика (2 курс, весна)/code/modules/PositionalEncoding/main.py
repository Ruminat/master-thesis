import math

import torch
import torch.nn as nn
from torch import Tensor


# helper Module that adds positional encoding to the token embedding
# to introduce a notion of word order
class PositionalEncoding(nn.Module):
  def __init__(
    self,
    embeddingSize: int,
    dropout: float,
    maxlen: int = 5000
  ):
    super(PositionalEncoding, self).__init__()
    den = torch.exp(-torch.arange(0, embeddingSize, 2) * math.log(10000) / embeddingSize)
    pos = torch.arange(0, maxlen).reshape(maxlen, 1)
    posEmbedding = torch.zeros((maxlen, embeddingSize))
    posEmbedding[:, 0::2] = torch.sin(pos * den)
    posEmbedding[:, 1::2] = torch.cos(pos * den)
    posEmbedding = posEmbedding.unsqueeze(-2)

    self.dropout = nn.Dropout(dropout)
    self.register_buffer("posEmbedding", posEmbedding)

  def forward(self, token_embedding: Tensor):
    return self.dropout(token_embedding + self.posEmbedding[:token_embedding.size(0), :])
