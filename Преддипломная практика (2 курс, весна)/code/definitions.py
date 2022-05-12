import torch

# The seed for PyTorch
SEED = 0
# Batch size for learning
BATCH_SIZE = 64

# Optimizer parameters
WEIGHT_DECAY = 0
LEARNING_RATE = 0.0001
BETAS = (0.9, 0.98)
EPSILON = 1e-9

DEVICE_CPU = "cpu"
DEVICE_GPU = "cuda"
# Which device to use for training/evaluation (uses CUDA when available, otherwise CPU)
DEVICE = torch.device(DEVICE_GPU if torch.cuda.is_available() else DEVICE_CPU)

# You can put a trained model into MODELS_DIR with file name DEFAULT_MODEL_FILENAME
# so you won't have to train it each time
MODELS_DIR = "./build"
DEFAULT_MODEL_FILENAME = "transformer.pt"
