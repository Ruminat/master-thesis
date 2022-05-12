import math

import torch.nn as nn
from torch import Tensor


# helper Module to convert tensor of input indices
# into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
  def __init__(self, vocabSize: int, embeddingSize):
    super(TokenEmbedding, self).__init__()
    self.embedding = nn.Embedding(vocabSize, embeddingSize)
    self.embeddingSize = embeddingSize

  def forward(self, tokens: Tensor):
    return self.embedding(tokens.long()) * math.sqrt(self.embeddingSize)
