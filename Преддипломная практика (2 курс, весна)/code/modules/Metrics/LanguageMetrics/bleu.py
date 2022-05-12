import torch
from datasets import load_metric
from modules.Metrics.definitions import TMetricsData
from modules.Metrics.utils import getSingleMetricScores

bleu = load_metric("bleu")

def getBleuScore(metricsData: TMetricsData) -> float:
  return bleu.compute(
    predictions=metricsData.translationTokens,
    references=metricsData.tgtTokens
  )

def getBleuScores(metricsData: TMetricsData) -> torch.Tensor:
  return getSingleMetricScores(getBleuScore, metricsData)
