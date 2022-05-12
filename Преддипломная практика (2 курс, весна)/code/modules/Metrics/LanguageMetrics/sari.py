from typing import List

import torch

from datasets import load_metric
from modules.Language.definitions import (TSentence, TTokensSentence,
                                          TTokensSentences)
from modules.Metrics.definitions import TMetricsData
from modules.Metrics.utils import getSingleMetricScores

sari = load_metric("sari")

def joinTokens(tokens: TTokensSentence) -> TSentence:
  return " ".join(tokens)

def parseTokensSentences(tokensList: TTokensSentences) -> TTokensSentence:
  return list(map(joinTokens, tokensList))

def parseTokensSentencesList(tokensList: List[TTokensSentences]) -> TTokensSentences:
  return list(map(parseTokensSentences, tokensList))

def getSariScore(metricsData: TMetricsData) -> float:
  return sari.compute(
    sources=parseTokensSentences(metricsData.srcTokens),
    predictions=parseTokensSentences(metricsData.translationTokens),
    references=parseTokensSentencesList(metricsData.tgtTokens)
  )["sari"]

def getSariScores(metricsData: TMetricsData) -> torch.Tensor:
  return getSingleMetricScores(getSariScore, metricsData)
