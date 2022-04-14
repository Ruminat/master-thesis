from datasets import load_metric
import torch
from torch.utils.data import DataLoader

from modules.Dataset.main import MyDataset

metric = load_metric("bleu")

def getBleuScore(model: torch.nn, dataset: MyDataset, srcKey: str, tgtKey: str):
  testSplit = dataset.getTestSplit()
  # collateFn = getCollateFn(model.srcLanguage, model.tgtLanguage, textTransform)
  testDataloader = DataLoader(testSplit)

  print("I HATE MYSELF", testDataloader)

  for datasetRow in testDataloader:
    print("SUKA", datasetRow)
    print("TVAR", datasetRow[srcKey])
    print("TVAR", datasetRow[tgtKey])
    # print("VOT SUKA", modelInputs)
    # print("VOT BLYAT", goldReferences)
    modelTranslation = model.translate(datasetRow[srcKey][0])
    metric.add_batch(predictions=modelTranslation, references=datasetRow[tgtKey][0])
  finalScore = metric.compute()
  return finalScore
  # return 0
