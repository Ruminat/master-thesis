from typing import Callable, List

import torch
import tqdm
from modules.Dataset.definitions import TJapaneseSimplificationDataset
from modules.Language.utils import formatSentence
from modules.Metrics.definitions import TMetricsData
from torch.utils.data import DataLoader


# Returns metrics data in the needed format so we can just call the metric functions with it as is
def getMetricsData(model: torch.nn, dataset: TJapaneseSimplificationDataset) -> TMetricsData:
  result = TMetricsData(
    srcSample = [],
    tgtSample = [],
    srcTokens = [],
    tgtTokens = [],
    translation = [],
    translationTokens = [],
  )
  testSplit = dataset.getTestSplit()
  testDataloader = DataLoader(testSplit)
  for datasetRow in tqdm.tqdm(testDataloader, leave=False):
    try:
      srcSample = formatSentence(datasetRow[dataset.srcSentenceKey][0])
      tgtSample = formatSentence(datasetRow[dataset.tgtSentenceKey][0])

      srcTokens = model.tokenize(srcSample)
      tgtTokens = model.tokenize(tgtSample)

      translation = model.translate(srcSample)
      if (srcSample[-1] == "。"):
        translation += srcSample[-1]
      translationTokens = model.tokenize(translation)

      result.srcSample.append(srcSample)
      result.tgtSample.append([tgtSample])
      result.srcTokens.append(srcTokens)
      result.tgtTokens.append([tgtTokens])
      result.translation.append(translation)
      result.translationTokens.append(translationTokens)
    except Exception as error:
      print("getMetricsData error:", error)
  return result

# Used for getting single metric values instead of one overall value
def getSingleMetricScores(
  metricFn: Callable[[TMetricsData], float],
  metricsData: TMetricsData
) -> torch.Tensor:
  result: List[float] = []
  for i in range(0, len(metricsData.srcSample)):
    singleMetricsData = TMetricsData(
      srcSample = [metricsData.srcSample[i]],
      tgtSample = [metricsData.tgtSample[i]],
      srcTokens = [metricsData.srcTokens[i]],
      tgtTokens = [metricsData.tgtTokens[i]],
      translation = [metricsData.translation[i]],
      translationTokens = [metricsData.translationTokens[i]],
    )
    result.append(metricFn(singleMetricsData))
  return torch.Tensor(result)
