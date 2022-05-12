from dataclasses import dataclass
from typing import Optional

from modules.Dataset.definitions import TJapaneseSimplificationDataset
from modules.Language.definitions import TTokenizer
from spacy import Vocab

from definitions import DEFAULT_MODEL_FILENAME, DEVICE


# The dataclass for Seq2SeqTransformer with adequate defaults
@dataclass
class TSeq2SeqTransformerParameters:
  dataset: TJapaneseSimplificationDataset
  maxEpochs: int = 30
  batchSize: int = 64
  attentionHeadsCount: int = 8
  encoderLayersCount: int = 6
  decoderLayersCount: int = 6
  embeddingSize: int = 512
  feedForwardSize: int = 512
  dropout: float = 0.1
  device: str = DEVICE
  fileName: str = DEFAULT_MODEL_FILENAME
  customTokenizer: Optional[TTokenizer] = None
  customVocab: Optional[Vocab] = None
