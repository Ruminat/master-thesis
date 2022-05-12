from modules.Language.definitions import SPACY_JP
from modules.Language.utils import getSpacyTokenizer, getVocabTransform
from modules.Seq2SeqTransformer.definitions import TSeq2SeqTransformerParameters
from modules.Dataset.snowSimplifiedJapanese.main import snowSimplifiedJapaneseDataset
from modules.Dataset.wikipediaJa.main import wikipediaJpDataset

jpSpacyTokenizer = getSpacyTokenizer(SPACY_JP)
jpVocabTransform = getVocabTransform(
  snowSimplifiedJapaneseDataset.srcSentenceKey,
  snowSimplifiedJapaneseDataset.tgtSentenceKey,
  jpSpacyTokenizer,
  snowSimplifiedJapaneseDataset
)

baseModelParams = TSeq2SeqTransformerParameters(
  dataset=snowSimplifiedJapaneseDataset
)

wikiModelParams = TSeq2SeqTransformerParameters(
  dataset=wikipediaJpDataset,
  customTokenizer=jpSpacyTokenizer,
  customVocab=jpVocabTransform,
  fileName="transformer-wiki.pt",
)

fromPretrainedParams = TSeq2SeqTransformerParameters(
  dataset=snowSimplifiedJapaneseDataset,
  fileName="from-pretrained.pt"
)
