from dataclasses import dataclass
from typing import Callable, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from modules.Language.definitions import SPACY_JP

# The dataset type (taken from the Huggingface datasets)
TDataset = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]

# A function that returns a dataset
# it's better to use functions which can be lazily loaded
# instead of loading every single dataset in advance
TDatasetFn = Callable[[], TDataset]

# The base data class for any dataset
@dataclass
class TDatasetBase:
  getTrainSplit: TDatasetFn
  getValidationSplit: Optional[TDatasetFn] = None
  getTestSplit: Optional[TDatasetFn] = None

  def iterateOverSplits(self):
    for row in self.getTrainSplit():
      yield row
    if (self.getValidationSplit is not None):
      for row in self.getValidationSplit():
        yield row
    if (self.getTestSplit is not None):
      for row in self.getTestSplit():
        yield row

# Simplification dataset data class
@dataclass
class TSimplificationDataset(TDatasetBase):
  srcSentenceKey: str = ""
  tgtSentenceKey: str = ""

# Japanese simplification dataset data class
@dataclass
class TJapaneseSimplificationDataset(TSimplificationDataset):
  spacyKey: str = SPACY_JP
