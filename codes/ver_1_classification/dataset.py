

import torch
from codes.ver_1_classification.param import Param
from codes.ver_1_classification.make_features import read_xvector, read_ivector

class _Dataset(torch.utils.data.Dataset):
  def __init__(self, data_dir):
    # X, Y = read_xvector(data_dir)
    X, Y = read_ivector(data_dir, 'all3')
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