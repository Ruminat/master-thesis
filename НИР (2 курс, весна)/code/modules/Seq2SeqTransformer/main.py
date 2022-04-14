from typing import List
import torch
import torch.nn as nn
from definitions import (SRC_LANGUAGE, TGT_LANGUAGE, textTransform,
                         vocabTransform)
from modules.Embedding.main import TokenEmbedding
from modules.Language.definitions import BOS_IDX, BOS_SYMBOL, EOS_SYMBOL
from modules.PositionalEncoding.main import PositionalEncoding
from modules.Seq2SeqTransformer.utils import greedyDecode
from torch import Tensor

from torch.nn import Transformer


# The final model to be trained
class Seq2SeqTransformer(nn.Module):
  def __init__(
    self,
    batchSize: int,
    srcLanguage: str,
    tgtLanguage: str,
    num_encoder_layers: int,
    num_decoder_layers: int,
    embeddingSize: int,
    nhead: int,
    srcVocabSize: int,
    tgtVocabSize: int,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    device: torch.device = torch.device("cpu")
  ):
    super(Seq2SeqTransformer, self).__init__()
    self.transformer = Transformer(
      d_model=embeddingSize,
      nhead=nhead,
      num_encoder_layers=num_encoder_layers,
      num_decoder_layers=num_decoder_layers,
      dim_feedforward=dim_feedforward,
      dropout=dropout
    )
    self.generator = nn.Linear(embeddingSize, tgtVocabSize)
    self.src_tok_emb = TokenEmbedding(srcVocabSize, embeddingSize)
    self.tgt_tok_emb = TokenEmbedding(tgtVocabSize, embeddingSize)
    self.positional_encoding = PositionalEncoding(embeddingSize, dropout=dropout)
    self.batchSize = batchSize
    self.srcLanguage = srcLanguage
    self.tgtLanguage = tgtLanguage
    self.device = device

  def forward(
    self,
    src: Tensor,
    trg: Tensor,
    srcMask: Tensor,
    tgtMask: Tensor,
    srcPaddingMask: Tensor,
    tgtPaddingMask: Tensor,
    memory_key_padding_mask: Tensor
  ):
    srcEmbedding = self.positional_encoding(self.src_tok_emb(src))
    tgtEmbedding = self.positional_encoding(self.tgt_tok_emb(trg))
    outs = self.transformer(
      srcEmbedding,
      tgtEmbedding,
      srcMask,
      tgtMask,
      None,
      srcPaddingMask,
      tgtPaddingMask,
      memory_key_padding_mask
    )
    return self.generator(outs)

  def encode(self, src: Tensor, srcMask: Tensor):
    return self.transformer.encoder(
      self.positional_encoding(self.src_tok_emb(src)),
      srcMask
    )

  def decode(self, tgt: Tensor, memory: Tensor, tgtMask: Tensor):
    return self.transformer.decoder(
      self.positional_encoding(self.tgt_tok_emb(tgt)),
      memory,
      tgtMask
    )

  # Method for translating from srcLanguage to tgtLangauge (Japanese -> simplified Japanese)
  def translate(self, srcSentence: str):
    self.eval()
    src = textTransform[SRC_LANGUAGE](srcSentence).view(-1, 1)
    num_tokens = src.shape[0]
    srcMask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgtTokens = greedyDecode(
      self,
      src,
      srcMask,
      maxLen=num_tokens + 5,
      startSymbol=BOS_IDX,
      device=self.device
    ).flatten()
    tokens = vocabTransform[TGT_LANGUAGE].lookup_tokens(list(tgtTokens.cpu().numpy()))

    return self.tokensToText(tokens)

  # Turns a list of tokens into a single string
  # ["what", "is", "love"] -> "what is love"
  def tokensToText(self, tokens: List[str]) -> str:
    result = ""
    for token in tokens:
      if token == BOS_SYMBOL or token == EOS_SYMBOL:
        continue
      if token == "。":
        break
      result += token
    return result
