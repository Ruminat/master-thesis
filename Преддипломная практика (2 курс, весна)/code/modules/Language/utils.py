from typing import Iterable, List

import torch
from modules.Dataset.main import MyDataset
from modules.Language.definitions import (BOS_IDX, EOS_IDX,
                                          LANGUAGE_TO_SPACY_DATASET, PAD_IDX,
                                          SPECIAL_SYMBOLS, UNK_IDX)
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# Returns a Spacy tokenizer for the given language
def getSpacyTokenizer(language: str):
  return get_tokenizer("spacy", language=LANGUAGE_TO_SPACY_DATASET[language])

# Create source and target language tokenizer. Make sure to install the dependencies.
# pip install -U spacy
# python -m spacy download ja_core_news_lg
def getTokenTransform(srcLanguage: str, tgtLanguage: str):
  return {
    srcLanguage: getSpacyTokenizer(srcLanguage),
    tgtLanguage: getSpacyTokenizer(tgtLanguage),
  }

def getVocabTransform(
  srcLanguage: str,
  tgtLanguage: str,
  tokenTransform: dict,
  dataset: MyDataset
):
  vocabTransform = {}
  for language in [srcLanguage, tgtLanguage]:
    # Training data Iterator
    trainIter = dataset.getTrainSplit()
    # Create torchtext"s Vocab object
    vocabTransform[language] = build_vocab_from_iterator(
      yieldTokens(tokenTransform, trainIter, language),
      min_freq=1,
      specials=SPECIAL_SYMBOLS,
      special_first=True
    )

  # Set UNK_IDX as the default index. This index is returned when the token is not found.
  # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
  for language in [srcLanguage, tgtLanguage]:
    vocabTransform[language].set_default_index(UNK_IDX)

  return vocabTransform

# Function to iterate through tokens
def yieldTokens(tokenTransform: dict, dataIter: Iterable, language: str) -> List[str]:
  tokenizer = tokenTransform[language]
  for dataSample in dataIter:
    sample = dataSample[language]
    yield tokenizer(sample)

# src and tgt language text transforms to convert raw strings into tensors indices
def getTextTransform(
  srcLanguage: str,
  tgtLanguage: str,
  tokenTransform: dict,
  vocabTransform: dict
):
  textTransform = {}
  for language in [srcLanguage, tgtLanguage]:
    textTransform[language] = sequentialTransforms(
      tokenTransform[language], # Tokenization
      vocabTransform[language], # Numericalization
      tensorTransform
    ) # Add BOS/EOS and create tensor
  return textTransform

# helper function to club together sequential operations
def sequentialTransforms(*transforms):
  def func(textInput: str):
    for transform in transforms:
      textInput = transform(textInput)
    return textInput
  return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensorTransform(tokenIds: List[int]):
  return torch.cat(
    (
      torch.tensor([BOS_IDX]),
      torch.tensor(tokenIds),
      torch.tensor([EOS_IDX])
    )
  )

# function to collate data samples into batch tensors
def getCollateFn(srcLanguage: str, tgtLanguage: str, textTransform: dict):
  def collateFn(batch):
    srcBatch, tgtBatch = [], []
    for sus in batch:
      srcSample, tgtSample = sus[srcLanguage], sus[tgtLanguage]
      srcBatch.append(textTransform[srcLanguage](srcSample.rstrip("\n")))
      tgtBatch.append(textTransform[tgtLanguage](tgtSample.rstrip("\n")))

    srcBatch = pad_sequence(srcBatch, padding_value=PAD_IDX)
    tgtBatch = pad_sequence(tgtBatch, padding_value=PAD_IDX)
    return srcBatch, tgtBatch
  return collateFn
