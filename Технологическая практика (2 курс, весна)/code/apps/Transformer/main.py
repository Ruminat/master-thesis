import sys

from definitions import DATASET
from modules.Language.definitions import JAPANESE_SIMPLIFIED, JAPANESE_SOURCE
from modules.Metrics.bleu import getBleuScore
from utils import (getTrainedTransformer, initiatePyTorch, loadTransformer,
                   prettyPrintTranslation)


def startTransformerApp():
  initiatePyTorch()

  if ("--train" in sys.argv):
    print("\n-- TRAIN MODE --\n")
    transformer = getTrainedTransformer()
  elif ("--load" in sys.argv):
    print("\n-- LOADING THE SAVED MODEL --\n")
    transformer = loadTransformer()
  else:
    print("\n-- DEFAULT (TRAIN) MODE --\n")
    transformer = getTrainedTransformer()

  # -- Testing the model --
  blueScore = getBleuScore(transformer, DATASET, JAPANESE_SOURCE, JAPANESE_SIMPLIFIED)
  print(f"BLEU score: {blueScore}")

  if ("--no-print" not in sys.argv):
    print("\nSentences that are not in the dataset\n")

    prettyPrintTranslation(transformer, "お前はもう死んでいる。")
    prettyPrintTranslation(transformer, "知識豊富な人間は実に馬鹿である。")
    prettyPrintTranslation(transformer, "あたしのこと好きすぎ。")
    prettyPrintTranslation(transformer, "事実上日本の唯一の公用語である。")
    prettyPrintTranslation(transformer, "我思う故に我あり。")

    print("\nSentences from the dataset\n")

    prettyPrintTranslation(transformer, "彼は怒りに我を忘れた。")
