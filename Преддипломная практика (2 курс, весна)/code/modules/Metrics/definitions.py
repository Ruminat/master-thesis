from dataclasses import dataclass
from typing import List

from modules.Language.definitions import TSentences, TTokensSentences


@dataclass
class TMetricsData:
  srcSample: TSentences
  tgtSample: List[TSentences]
  srcTokens: TTokensSentences
  tgtTokens: List[TTokensSentences]
  translation: TSentences
  translationTokens: TTokensSentences
