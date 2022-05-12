import functools

from datasets import load_dataset
from modules.Dataset.definitions import TJapaneseSimplificationDataset
from modules.Dataset.snowSimplifiedJapanese.main import getValidationSplit

fileName = "modules/Dataset/wikipediaJa/data/wikipediaJp.csv"

@functools.cache
def getTrainSplit():
  dataset = load_dataset("csv", data_files=fileName, split=f"train")
  print("loaded the train split (wikiJa)", dataset.num_rows)
  return dataset

wikipediaJpDataset = TJapaneseSimplificationDataset(
  getTrainSplit=getTrainSplit,
  getValidationSplit=getValidationSplit,
  srcSentenceKey="original_ja",
  tgtSentenceKey="simplified_ja"
)
