from typing import List

import torch
import torch.nn as nn
from modules.Embedding.main import TokenEmbedding
from modules.Language.definitions import (BOS_IDX, BOS_SYMBOL, EOS_SYMBOL,
                                          TTokenizer)
from modules.Language.utils import tensorTransform
from modules.PositionalEncoding.main import PositionalEncoding
from modules.Seq2SeqTransformer.definitions import \
    TSeq2SeqTransformerParameters
from modules.Seq2SeqTransformer.utils import greedyDecode
from spacy import Vocab
from torch import Tensor
from torch.nn import Transformer


# The final model to be trained
class Seq2SeqTransformer(nn.Module):
  def __init__(
    self,
    params: TSeq2SeqTransformerParameters,
    tokenize: TTokenizer,
    vocab: dict[str, Vocab]
  ):
    super(Seq2SeqTransformer, self).__init__()
    self.transformer = Transformer(
      d_model=params.embeddingSize,
      nhead=params.attentionHeadsCount,
      num_encoder_layers=params.encoderLayersCount,
      num_decoder_layers=params.decoderLayersCount,
      dim_feedforward=params.feedForwardSize,
      dropout=params.dropout
    )

    self.srcLanguage = params.dataset.srcSentenceKey
    self.tgtLanguage = params.dataset.tgtSentenceKey
    self.tokenize = tokenize
    self.vocab = vocab

    self.batchSize = params.batchSize
    self.device = params.device

    srcVocabSize = len(vocab[self.srcLanguage])
    tgtVocabSize = len(vocab[self.tgtLanguage])

    self.generator = nn.Linear(params.embeddingSize, tgtVocabSize)
    self.srcEmbedding = TokenEmbedding(srcVocabSize, params.embeddingSize)
    self.tgtEmbedding = TokenEmbedding(tgtVocabSize, params.embeddingSize)
    self.positionalEncoding = PositionalEncoding(params.embeddingSize, dropout=params.dropout)

    print(f"""
      Created a Transformer model {self.srcLanguage} -> {self.tgtLanguage}:
        - device "{self.device}"
        - batch size = {self.batchSize}
        - vocab ({srcVocabSize} -> {tgtVocabSize})
    """)

  def srcTextTransform(self, text: str) -> Tensor:
    return tensorTransform(self.vocab[self.srcLanguage](self.tokenize(text)))

  def tgtTextTransform(self, text: str) -> Tensor:
    return tensorTransform(self.vocab[self.tgtLanguage](self.tokenize(text)))

  def forward(
    self,
    src: Tensor,
    tgt: Tensor,
    srcMask: Tensor,
    tgtMask: Tensor,
    srcPaddingMask: Tensor,
    tgtPaddingMask: Tensor,
    memoryKeyPaddingMask: Tensor
  ):
    srcEmbedding = self.positionalEncoding(self.srcEmbedding(src))
    tgtEmbedding = self.positionalEncoding(self.tgtEmbedding(tgt))
    output = self.transformer(
      srcEmbedding,
      tgtEmbedding,
      srcMask,
      tgtMask,
      None,
      srcPaddingMask,
      tgtPaddingMask,
      memoryKeyPaddingMask
    )
    return self.generator(output)

  def encode(self, src: Tensor, srcMask: Tensor):
    return self.transformer.encoder(
      self.positionalEncoding(self.srcEmbedding(src)),
      srcMask
    )

  def decode(self, tgt: Tensor, memory: Tensor, tgtMask: Tensor):
    return self.transformer.decoder(
      self.positionalEncoding(self.tgtEmbedding(tgt)),
      memory,
      tgtMask
    )

  # Method for translating from srcLanguage to tgtLangauge (Japanese -> simplified Japanese)
  def translate(self, srcSentence: str):
    self.eval()
    transformedSentence = self.srcTextTransform(srcSentence)
    src = transformedSentence.view(-1, 1)
    numTokens = src.shape[0]
    srcMask = (torch.zeros(numTokens, numTokens)).type(torch.bool)
    tgtTokens = greedyDecode(
      self,
      src,
      srcMask,
      maxLen=numTokens + 5,
      startSymbol=BOS_IDX,
      device=self.device
    )
    tgtTokens = tgtTokens.flatten()
    tokensToLookUp = list(tgtTokens.cpu().numpy())
    tokens = self.vocab[self.tgtLanguage].lookup_tokens(tokensToLookUp)

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
