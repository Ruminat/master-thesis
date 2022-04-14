from timeit import default_timer as timer

import torch
from modules.Dataset.main import MyDataset
from modules.Language.definitions import BOS_IDX, PAD_IDX
from modules.Language.utils import getCollateFn
from torch import Tensor, nn
from torch.utils.data import DataLoader


# Generates the following mask:
# tensor([[0., -inf, -inf, -inf, -inf],
#         [0.,   0., -inf, -inf, -inf],
#         [0.,   0.,   0., -inf, -inf],
#         [0.,   0.,   0.,   0., -inf],
#         [0.,   0.,   0.,   0.,   0.]])
def generateSquareSubsequentMask(size: int, device: str) -> Tensor:
  mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float("-inf"))
  mask = mask.masked_fill(mask == 1, float(0.0))
  return mask

# Generates the seq2seq transformer masks
def createMask(src: Tensor, tgt: Tensor, device: str):
  src_seq_len = src.shape[0]
  tgt_seq_len = tgt.shape[0]

  tgt_mask = generateSquareSubsequentMask(tgt_seq_len, device)
  src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

  src_padding_mask = (src == PAD_IDX).transpose(0, 1)
  tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
  return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# Initializes the Transformer's parameters with the Glorot initialization.
def initializeTransformerParameters(transformer: nn.Module) -> None:
  for parameter in transformer.parameters():
    if parameter.dim() > 1:
      nn.init.xavier_uniform_(parameter)

# Function to generate output sequence (simplified sentence) using the greedy algorithm.
def greedyDecode(model, src, srcMask, maxLen, startSymbol, device: str) -> Tensor:
  src = src.to(device)
  srcMask = srcMask.to(device)

  memory = model.encode(src, srcMask)
  ys = torch.ones(1, 1).fill_(startSymbol).type(torch.long).to(device)
  for i in range(maxLen - 1):
    memory = memory.to(device)
    tgtMaskBase = generateSquareSubsequentMask(ys.size(0), device)
    tgtMask = (tgtMaskBase.type(torch.bool)).to(device)
    out = model.decode(ys, memory, tgtMask)
    out = out.transpose(0, 1)
    prob = model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.item()

    ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
    if next_word == BOS_IDX:
      break
  return ys

# Evaluates the model.
def evaluate(
  model: torch.nn,
  lossFn: torch.nn,
  textTransform: dict,
  dataset: MyDataset
) -> float:
  model.eval()
  losses = 0

  valIter = dataset.getValidationSplit()
  collateFn = getCollateFn(model.srcLanguage, model.tgtLanguage, textTransform)
  valDataloader = DataLoader(valIter, batch_size=model.batchSize, collate_fn=collateFn)

  for src, tgt in valDataloader:
    src = src.to(model.device)
    tgt = tgt.to(model.device)

    tgtInput = tgt[:-1, :]
    srcMask, tgtMask, srcPaddingMask, tgtPaddingMask = createMask(
      src,
      tgtInput,
      device=model.device
    )
    logits = model(
      src,
      tgtInput,
      srcMask,
      tgtMask,
      srcPaddingMask,
      tgtPaddingMask,
      srcPaddingMask
    )

    tgt_out = tgt[1:, :]
    loss = lossFn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    losses += loss.item()

  return losses / len(valDataloader)

# Trains the model
def train(
  model: torch.nn,
  optimizer: torch.optim,
  lossFn: torch.nn,
  textTransform: dict,
  epochs: int,
  dataset: MyDataset
) -> None:
  bestValue = 1729
  bestValueEpoch = 1

  for epoch in range(1, epochs + 1):
    startTime = timer()
    trainLoss = trainEpoch(model, optimizer, lossFn, textTransform, dataset)
    endTime = timer()
    trainTime = endTime - startTime

    startTime = timer()
    valueLoss = evaluate(model, lossFn, textTransform, dataset)
    endTime = timer()
    evaluationTime = endTime - startTime

    trainLossPrint = f"train loss: {trainLoss:.3f} ({trainTime:.1f}s)"
    valLossPrint = f"val loss: {valueLoss:.3f} ({evaluationTime:.1f})"
    print((f"epoch-{epoch}: ${trainLossPrint}, ${valLossPrint}"))

    if (valueLoss < bestValue):
      bestValue = valueLoss
      bestValueEpoch = epoch
    if (epoch - bestValueEpoch > 3):
      print("The model stopped improving, so we stop the learning process.")
      break

# One train epoch
def trainEpoch(
  model: torch.nn,
  optimizer: torch.optim,
  lossFn: torch.nn,
  textTransform: dict,
  dataset: MyDataset
) -> float:
  model.train()
  losses = 0
  train_iter = dataset.getTrainSplit()
  collateFn = getCollateFn(model.srcLanguage, model.tgtLanguage, textTransform)
  trainSplit = DataLoader(train_iter, batch_size=model.batchSize, collate_fn=collateFn)

  for src, tgt in trainSplit:
    src = src.to(model.device)
    tgt = tgt.to(model.device)

    tgtInput = tgt[:-1, :]
    srcMask, tgtMask, srcPaddingMask, tgtPaddingMask = createMask(
      src,
      tgtInput,
      device=model.device
    )
    logits = model(
      src,
      tgtInput,
      srcMask,
      tgtMask,
      srcPaddingMask,
      tgtPaddingMask,
      srcPaddingMask
    )

    optimizer.zero_grad()

    tgt_out = tgt[1:, :]
    loss = lossFn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    loss.backward()

    optimizer.step()
    losses += loss.item()

  return losses / len(trainSplit)
