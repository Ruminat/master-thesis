import torch

from modules.Dataset.snowSimplifiedJapanese.main import \
    snowSimplifiedJapaneseDataset
from modules.Language.definitions import JAPANESE_SIMPLIFIED, JAPANESE_SOURCE
from modules.Language.utils import (getTextTransform, getTokenTransform,
                                    getVocabTransform)


# You can put a trained model into MODELS_DIR with file name DEFAULT_MODEL_FILENAME
# so you won't have to train it each time
MODELS_DIR = "./build"
DEFAULT_MODEL_FILENAME = "transformer.pt"

# Source and target languages for translation (Japanese and simplified Japanese)
SRC_LANGUAGE = JAPANESE_SOURCE
TGT_LANGUAGE = JAPANESE_SIMPLIFIED

# The dataset we're gonna train the model on
DATASET = snowSimplifiedJapaneseDataset

# Transforms for the source sentence (text -> embeddings)
tokenTransform = getTokenTransform(SRC_LANGUAGE, TGT_LANGUAGE)
vocabTransform = getVocabTransform(SRC_LANGUAGE, TGT_LANGUAGE, tokenTransform, DATASET)
textTransform = getTextTransform(SRC_LANGUAGE, TGT_LANGUAGE, tokenTransform, vocabTransform)

# The seed for PyTorch
SEED = 0
# Batch size for learning
BATCH_SIZE = 64

# Optimizer parameters
WEIGHT_DECAY = 0
LEARNING_RATE = 0.0001
BETAS = (0.9, 0.98)
EPSILON = 1e-9

# The number of training epochs
NUM_EPOCHS = 30
# The size of the embedding vectors
EMB_SIZE = 512
# The number of attention heads
NHEAD = 8
# Size of the feed forward layer
DIM_FEEDFORWARD = 512
# The number of encoder/decoder layers
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = NUM_ENCODER_LAYERS

# Supplementary constants
SRC_VOCAB_SIZE = len(vocabTransform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocabTransform[TGT_LANGUAGE])
DEVICE_CPU = "cpu"
DEVICE_GPU = "cuda"
# Which device to use for training/evaluation (uses CUDA when available, otherwise CPU)
DEVICE = torch.device(DEVICE_GPU if torch.cuda.is_available() else DEVICE_CPU)
