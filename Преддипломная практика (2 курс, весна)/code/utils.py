from typing import List
import torch

from definitions import (BETAS, DEFAULT_MODEL_FILENAME, DEVICE, EPSILON,
                         LEARNING_RATE, MODELS_DIR, SEED, WEIGHT_DECAY)
from modules.Language.definitions import PAD_IDX
from modules.Language.utils import getSpacyTokenizer, getVocabTransform
from modules.Seq2SeqTransformer.definitions import \
    TSeq2SeqTransformerParameters
from modules.Seq2SeqTransformer.main import Seq2SeqTransformer
from modules.Seq2SeqTransformer.utils import (initializeTransformerParameters,
                                              train)


# Initiate PyTorch with seed and clear CUDA cache
def initiatePyTorch() -> None:
  torch.manual_seed(SEED)
  torch.cuda.empty_cache()
  print(f"Running PyTorch on {DEVICE} with seed={SEED}")

# Pretty prints a list of sentences
def prettyPrintSentencesTranslation(transformer: Seq2SeqTransformer, sentences: List[str]) -> None:
  for sentence in sentences:
    prettyPrintTranslation(transformer, sentence)

# Pretty prints translation as
# Translating: «source» -> «simplified_version»
def prettyPrintTranslation(transformer: Seq2SeqTransformer, sourceSentence: str) -> None:
  src = f"«{sourceSentence.strip()}»"
  result = f"«{transformer.translate(sourceSentence).strip()}»"
  print("Translating:", src, "->", result)

# Loads a transformer model from the ./build directory
def loadTransformer(fileName: str = DEFAULT_MODEL_FILENAME) -> Seq2SeqTransformer:
  print("Loading Transformer...")
  transformer = torch.load(f"{MODELS_DIR}/{fileName}")
  transformer.eval()
  print("Transformer is ready to use")
  return transformer

# Trains a transformer model with the given parameters and returns it
def getTrainedTransformer(params: TSeq2SeqTransformerParameters) -> Seq2SeqTransformer:
  tokenize = params.customTokenizer \
    if params.customTokenizer is not None \
    else getSpacyTokenizer(params.dataset.spacyKey)
  vocab = params.customVocab \
    if params.customVocab is not None \
    else getVocabTransform(
      params.dataset.srcSentenceKey,
      params.dataset.tgtSentenceKey,
      tokenize,
      params.dataset
    )

  transformer = Seq2SeqTransformer(params=params, tokenize=tokenize, vocab=vocab)
  print("Created the Transformer model")
  initializeTransformerParameters(transformer)
  print("Initialized parameters")

  transformer = transformer.to(DEVICE)
  lossFn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
  optimizer = torch.optim.Adam(
    transformer.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=BETAS,
    eps=EPSILON
  )
  print("Created lossFn and optimizer")

  print("Training the model...")
  train(transformer, optimizer, lossFn, params.maxEpochs, params.dataset)
  print("The model has trained well")

  torch.save(transformer, f"{MODELS_DIR}/{params.fileName}")

  return transformer

# Trains a Transformer model from a pretrained Transformer
# with lowering the Learning Rate for the encoder layer
def fromPretrained(
  transformer: Seq2SeqTransformer,
  params: TSeq2SeqTransformerParameters
) -> Seq2SeqTransformer:
  encoder = []
  rest = []
  for name, param in transformer.named_parameters():
    if "encoder" in name:
      encoder.append(param)
    else:
      rest.append(param)

  lossFn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
  optimizer = torch.optim.Adam(
    [{'params': encoder}, {'params': rest}],
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=BETAS,
    eps=EPSILON
  )
  optimizer.param_groups[0]['lr'] = LEARNING_RATE / 5
  train(transformer, optimizer, lossFn, params.maxEpochs, params.dataset)
  torch.save(transformer, f"{MODELS_DIR}/{params.fileName}")
  return transformer
