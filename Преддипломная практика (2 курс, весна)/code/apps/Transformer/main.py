import sys

from apps.Transformer.definitions import (baseModelParams,
                                          fromPretrainedParams,
                                          wikiModelParams)
from modules.Dataset.snowSimplifiedJapanese.main import \
    snowSimplifiedJapaneseDataset
from modules.Metrics.LanguageMetrics.bleu import getBleuScore
from modules.Metrics.LanguageMetrics.sari import getSariScore
from modules.Metrics.utils import getMetricsData
from modules.Seq2SeqTransformer.main import Seq2SeqTransformer
from utils import (fromPretrained, getTrainedTransformer, initiatePyTorch,
                   loadTransformer, prettyPrintSentencesTranslation)


def startTransformerApp() -> None:
  initiatePyTorch()

  if ("--pretrain" in sys.argv):
    print("\n-- PRETRAIN MODE --\n")
    transformer = getTrainedTransformer(wikiModelParams)
  elif ("--from-pretrained" in sys.argv):
    print("\n-- FROM PRETRAINED MODE --\n")
    pretrainedTransformer = loadTransformer(wikiModelParams.fileName)
    transformer = fromPretrained(pretrainedTransformer, fromPretrainedParams)
  elif ("--train" in sys.argv):
    print("\n-- TRAIN MODE --\n")
    transformer = getTrainedTransformer(baseModelParams)
  elif ("--load" in sys.argv):
    print("\n-- LOADING THE SAVED MODEL --\n")
    transformer = loadTransformer(baseModelParams.fileName)
  else:
    print("""
      Couldn't parse the provided command.
      You can type
        `python main.py --help`
      to get the list of available commands.
    """)

  if ("--no-print" not in sys.argv):
    # -- Testing the model --
    printTransformerTests(transformer)

def printTransformerTests(transformer: Seq2SeqTransformer) -> None:
  print("\nSentences that are not in the dataset\n")

  prettyPrintSentencesTranslation(transformer, [
    "お前はもう死んでいる。",
    "知識豊富な人間は実に馬鹿である。",
    "あたしのこと好きすぎ。",
    "事実上日本の唯一の公用語である。",
    "我思う故に我あり。",
  ])

  print("\nSentences from the dataset\n")

  prettyPrintSentencesTranslation(transformer, [
    "彼は怒りに我を忘れた。",
    "ジョンは今絶頂だ。",
    "ビル以外はみなあつまった。",
    "彼はすぐに風邪をひく。",
    "彼は私のしたことにいちいち文句を言う。",
    "彼女は内気なので、ますます彼女が好きだ。",
    "英仏海峡を泳ぎ渡るのに成功した最初の人はウェッブ船長でした。",
    "彼女はほほえんで僕のささやかなプレゼントを受け取ってくれた。",
    "料理はそうおいしくはなかったけれど、その他の点ではパーティーは成功した。",
    "その絵の値段は１０ポンドです。","その絵の価格はポンドです。",
  ])

  print("\nMetrics\n")

  print("Calculating the metrics data...")
  metricsData = getMetricsData(transformer, snowSimplifiedJapaneseDataset)
  print("Calculating the metrics...")
  blueScore = getBleuScore(metricsData)
  print(f"BLEU score: {blueScore}")
  sariScore = getSariScore(metricsData)
  print(f"SARI score: {sariScore}")
