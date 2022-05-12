from typing import Callable, List

from torch import Tensor

# Helper types
TToken = str
TTokensSentence = List[TToken]
TTokensSentences = List[TTokensSentence]

TSentence = str
TSentences = List[TSentence]

TTokenizer = Callable[[TSentence], TTokensSentence]
TTextTransformer = Callable[[TSentence], Tensor]

# Special symbols
UNK_SYMBOL = "<unk>" # unknown symbol
PAD_SYMBOL = "<pad>" # padding symbol
BOS_SYMBOL = "<bos>" # «Begginning Of a Sentence» symbol
EOS_SYMBOL = "<eos>" # «End Of a Sentence» symbol

# Indicies for special symbols
UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
SPECIAL_SYMBOLS = [UNK_SYMBOL, PAD_SYMBOL, BOS_SYMBOL, EOS_SYMBOL]

# Spacy dataset for Japanese
SPACY_JP = "ja_core_news_lg"
