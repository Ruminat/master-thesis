import functools

from datasets import concatenate_datasets, load_dataset, logging
from modules.Dataset.definitions import TJapaneseSimplificationDataset

# Disabling the logging because it's useless and annoying
logging.set_verbosity_error()

SNOW_DATASET = "snow_simplified_japanese_corpus"

SNOW_T15 = "snow_t15"
SNOW_T23 = "snow_t23"

@functools.cache
def getTrainSplit():
  snow15Dataset = load_dataset(SNOW_DATASET, SNOW_T15, split=f"train")
  snow23Dataset = load_dataset(SNOW_DATASET, SNOW_T23, split=f"train[2000:]")
  dataset = concatenate_datasets([snow15Dataset, snow23Dataset])
  print("loaded the train split (SNOW)", dataset.num_rows)
  return dataset

@functools.cache
def getValidationSplit():
  dataset = load_dataset(SNOW_DATASET, SNOW_T23, split=f"train[:1000]")
  print("loaded the validation split (SNOW)", dataset.num_rows)
  return dataset

@functools.cache
def getTestSplit():
  dataset = load_dataset(SNOW_DATASET, SNOW_T23, split=f"train[1000:2000]")
  print("loaded the test split (SNOW)", dataset.num_rows)
  return dataset

snowSimplifiedJapaneseDataset = TJapaneseSimplificationDataset(
  getTrainSplit=getTrainSplit,
  getValidationSplit=getValidationSplit,
  getTestSplit=getTestSplit,
  srcSentenceKey="original_ja",
  tgtSentenceKey="simplified_ja"
)
