
import time
import torch
import torch.nn as nn
from pa_nlp.nlp import Logger
from pa_nlp.pytorch.estimator.train import TrainerBase
from codes.ver_1_classification.param import Param
from codes.ver_1_classification.model import Model
from codes.ver_1_classification.dataset import get_batch_data

@torch.no_grad()
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
  """
  if smoothing == 0, it's one-hot method
  if 0 < smoothing < 1, it's smooth method
  Warning: This function has no grad.
  """
  # assert 0 <= smoothing < 1
  confidence = 1.0 - smoothing
  label_shape = torch.Size((true_labels.size(0), classes))

  smooth_label = torch.empty(size=label_shape, device=true_labels.device)
  smooth_label.fill_(smoothing / (classes - 1))
  # print(smooth_label)
  # smooth_label.scatter_(1, true_labels.data.unsqueeze(1), confidence)
  return smooth_label

class LabelSmoothingLoss(nn.Module):
  """This is label smoothing loss function.
  """

  def __init__(self, classes, smoothing=0.0, dim=-1):
    super(LabelSmoothingLoss, self).__init__()
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing
    self.cls = classes
    self.dim = dim

  def forward(self, pred, target):
    pred = pred.log_softmax(dim=self.dim)
    true_dist = smooth_one_hot(target, self.cls, self.smoothing)
    return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class Trainer(TrainerBase):
  def __init__(self):
    param = Param()
    param.verify()

    self._opt_vali_error = -100
    model = Model()

    optimizer_parameters = [
      {
        'params': [
          p for n, p in model.named_parameters()
        ],
        'lr': param.lr,
      },
    ]

    optimizer = getattr(torch.optim, param.optimizer_name)(
      optimizer_parameters
    )

    super(Trainer, self).__init__(
      param, model,
      get_batch_data(param.train_files, param.epoch_num, True),
      optimizer
    )
    self.label_smoothing_loss = LabelSmoothingLoss(self._param.class_num,
                                              self._param.label_smoothing)

  def _train_one_batch(self, x, y):
    logits = self.predict(x)
    logits = logits.view(-1, self._param.class_num)
    gold = y.view(-1)
    if self._param.label_smoothing:
      targets = smooth_one_hot(gold, self._param.class_num, self._param.label_smoothing)
      loss = self.label_smoothing_loss(logits, targets)
    else:
      loss = nn.functional.cross_entropy(logits, gold, reduction="mean")
    return loss

  def predict(self, x):
    logit = self._model(x)
    return logit

  def evaluate_file(self, data_dir):
    start_time = time.time()
    all_true_labels = []
    all_pred_labels = []
    for _, batch in get_batch_data(data_dir, 1, False):
      batch = [e.to(self._device) for e in batch]
      x, y = batch
      logits = self.predict(x)
      logits = logits.view(-1, self._param.class_num)
      pred_labels = logits.max(1)[1]
      all_true_labels.extend(y.tolist())
      all_pred_labels.extend(pred_labels.tolist())

    result = sum([x!=y for x,y in zip(all_true_labels, all_pred_labels)])/len(all_true_labels)
    total_time = time.time() - start_time
    avg_time = total_time / (len(all_true_labels) + 1e6)
    Logger.info(
      f"eval[{self._run_sample_num}]: "
      f"file={data_dir} wer={result} "
      f"total_time={total_time:.4f} secs avg_time={avg_time:.4f} sec/sample "
    )

    return result

if __name__ == '__main__':
  trainer = Trainer()
  trainer.train()
