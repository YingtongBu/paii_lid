
import os
import time
import torch
import torch.nn as nn
from pa_nlp.nlp import Logger
from pa_nlp.pytorch.estimator.param import ParamBase
from pa_nlp.pytorch.estimator.train import TrainerBase


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.hidden1 = nn.Linear(512, 1024)
    self.hidden2 = nn.Linear(1024, 1024)
    self.hidden3 = nn.Linear(1024, 2)

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

def read_xvector(filepath):
  X = []
  y = []
  files = os.listdir(filepath)
  for file in files:
    with open(f'{filepath}/{file}', 'r') as f:
      for line in f:
        item = line.strip().split()
        utt = item[0]
        label = utt.split('-')[0]
        if label == 'us':
          label = 0
        elif label == 'uk':
          label = 1
        else:
          continue
        vector = [float(x) for x in item[1:][1:-1]]
        X.append(vector)
        y.append(label)
  return X, y

class _Dataset(torch.utils.data.Dataset):
  def __init__(self, data_dir):
    X, Y = read_xvector(data_dir)
    data = [(x, y)  for x, y in zip(X, Y)]
    self._data = data

  def __len__(self):
    return len(self._data)

  def __getitem__(self, idx):
    return self._data[idx]

def _pad_batch_data(batch):
  batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
  x, y = list(zip(*batch))
  x = torch.FloatTensor(x)
  y = torch.LongTensor(y)
  return x, y

def get_batch_data(data_dir, epoch_num, shuffle: bool):
  param = Param()
  dataset = _Dataset(data_dir)
  data_iter = torch.utils.data.DataLoader(
    dataset, param.get_batch_size_all_gpus(), shuffle=shuffle,
    num_workers=2,
    collate_fn=lambda x: x
  )

  for epoch_id in range(epoch_num):
    for batch in data_iter:
      yield epoch_id, _pad_batch_data(batch)


class Param(ParamBase):
  def __init__(self):
    super(Param, self).__init__("neural_network")

    self.train_dir = 'xvectors_train.2'
    self.vali_dir = 'xvectors_test.2'

    self.gpus = []
    self.batch_dim = 0
    self.optimizer_name = "Adam"
    self.lr = 5e-4
    self.epoch_num   = 20
    self.batch_size_one_gpu  = 32
    self.batch_size_inference_one_gpu = 32

    self.eval_gap_instance_num = 1000
    self.train_sample_num = 180000
    self.use_polynormial_decay = True
    self.incremental_train = False

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
      get_batch_data(param.train_dir, param.epoch_num, True),
      optimizer
    )

  def _train_one_batch(self, x, y):
    print(x.size())
    logits = self.predict(x)
    logits = logits.view(-1, 2)
    gold = y.view(-1)
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
      logits = logits.view(-1, 2)
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

'export PYTHONPATH=$PYTHONPATH:/data/pytong/nlp_team:/data/pytong'
# modified nlp.param.verify() three lines about vali_file