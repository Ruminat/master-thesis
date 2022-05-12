from typing import Iterable, List

import torch
from modules.Dataset.definitions import TJapaneseSimplificationDataset
from modules.Language.definitions import (BOS_IDX, EOS_IDX, PAD_IDX, SPACY_JP,
                                          SPECIAL_SYMBOLS, UNK_IDX, TTextTransformer, TTokenizer)
from spacy import Vocab
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# Create source and target language tokenizer. Make sure to install the dependencies
# pip install -U spacy
# python -m spacy download ja_core_news_lg
def getTokenTransform(srcLanguage: str, tgtLanguage: str) -> dict[str, TTokenizer]:
  return {
    srcLanguage: getSpacyTokenizer(srcLanguage),
    tgtLanguage: getSpacyTokenizer(tgtLanguage),
  }

# Returns a Spacy tokenizer for the given language
def getSpacyTokenizer(language: str) -> TTokenizer:
  print(f"Creating a spacy tokenizer ({language})...")
  return get_tokenizer("spacy", language=SPACY_JP)

# Get source and target Vocab's
def getVocabTransform(
  srcLanguage: str,
  tgtLanguage: str,
  tokenize: TTokenizer,
  dataset: TJapaneseSimplificationDataset
) -> dict[str, Vocab]:
  return {
    srcLanguage: getVocab(tokenize, dataset, srcLanguage),
    tgtLanguage: getVocab(tokenize, dataset, tgtLanguage),
  }

# Creates a Vocab object for the given tokenize function and dataset
def getVocab(
  tokenize: TTokenizer,
  dataset: TJapaneseSimplificationDataset,
  language: str
) -> Vocab:
  print(f"Creating Vocab ({language})...")
  dataIter = dataset.getTrainSplit()
  # Create torchtext's Vocab object
  vocabTransform = build_vocab_from_iterator(
    yieldTokens(tokenize, dataIter, language),
    min_freq=1,
    specials=SPECIAL_SYMBOLS,
    special_first=True
  )

  # Set UNK_IDX as the default index. This index is returned when the token is not found.
  # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
  vocabTransform.set_default_index(UNK_IDX)

  return vocabTransform

# Function to iterate through tokens
def yieldTokens(tokenize: TTokenizer, dataIter: Iterable, language: str) -> List[str]:
  for dataSample in dataIter:
    yield tokenize(dataSample[language])

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
def getCollateFn(
  srcLanguage: str,
  tgtLanguage: str,
  srcTextTransform: TTextTransformer,
  tgtTextTransform: TTextTransformer
):
  def collateFn(batch):
    srcBatch, tgtBatch = [], []
    for sample in batch:
      srcSample, tgtSample = sample[srcLanguage], sample[tgtLanguage]
      srcBatch.append(srcTextTransform(formatSentence(srcSample)))
      tgtBatch.append(tgtTextTransform(formatSentence(tgtSample)))

    srcBatch = pad_sequence(srcBatch, padding_value=PAD_IDX)
    tgtBatch = pad_sequence(tgtBatch, padding_value=PAD_IDX)
    return srcBatch, tgtBatch
  return collateFn

# Helper for removing redundant whitespace and dot symbols
def formatSentence(sentence: str, removeDot = False) -> str:
  if (removeDot):
    return sentence.rstrip("。").rstrip("\n")
  else:
    return sentence.rstrip("\n")
