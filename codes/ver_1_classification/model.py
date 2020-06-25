

import torch.nn as nn
from pa_nlp.nlp import Logger


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.hidden1 = nn.Linear(600, 1024)
    self.hidden2 = nn.Linear(1024, 1024)
    self.hidden3 = nn.Linear(1024, 3)

  def _init_weights(self):
    for name, w in self.named_parameters():
      if "roberta" in name:
        continue

      if "dense" in name:
        if "weight" in name:
          nn.init.xavier_normal_(w)
        elif "bias" in name:
          nn.init.zeros_(w)
        else:
          Logger.warn(f"Unintialized '{name}'")

      else:
        Logger.warn(f"Unintialized '{name}'")

  def forward(self, x):
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.hidden3(x)
    return x
