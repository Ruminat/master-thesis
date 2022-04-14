from typing import Callable, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

TDataset = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]

TDatasetFn = Callable[[], TDataset]
