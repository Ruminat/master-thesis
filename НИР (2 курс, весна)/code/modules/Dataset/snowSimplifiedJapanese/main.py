import functools
from datasets import concatenate_datasets, load_dataset

from modules.Dataset.main import MyDataset

SNOW_DATASET = "snow_simplified_japanese_corpus"

SNOW_T15 = "snow_t15"
SNOW_T23 = "snow_t23"

VALIDATION_PERCENT = 5
TEST_PERCENT = 5

@functools.cache
def getTrainSplit():
  t15Dataset = load_dataset(SNOW_DATASET, SNOW_T15, split=f"train[{VALIDATION_PERCENT}%:]")
  t23Dataset = load_dataset(SNOW_DATASET, SNOW_T23, split=f"train[{TEST_PERCENT}%:]")
  return concatenate_datasets([t15Dataset, t23Dataset])

@functools.cache
def getValidationSplit():
  return load_dataset(SNOW_DATASET, SNOW_T15, split=f"train[:{VALIDATION_PERCENT}%]")

@functools.cache
def getTestSplit():
  return load_dataset(SNOW_DATASET, SNOW_T23, split=f"train[:{TEST_PERCENT}%]")

snowSimplifiedJapaneseDataset = MyDataset(
  getTrainSplit=getTrainSplit,
  getValidationSplit=getValidationSplit,
  getTestSplit=getTestSplit
)
