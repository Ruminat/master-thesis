from typing import Optional
from modules.Dataset.definitions import TDatasetFn


class MyDataset:
  def __init__(
    self,
    getTrainSplit: TDatasetFn,
    getValidationSplit: Optional[TDatasetFn],
    getTestSplit: Optional[TDatasetFn]
  ):
    self.getTrainSplit = getTrainSplit
    self.getValidationSplit = getValidationSplit
    self.getTestSplit = getTestSplit
