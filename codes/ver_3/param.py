import os
from pa_nlp.pytorch.estimator.param import ParamBase


class Param(ParamBase):
  def __init__(self):
    super(Param, self).__init__("ver_3")

    self.train_files = [
      f"{self.path_feat}/train.pydict"
    ]
    self.vali_file = f"{self.path_feat}/vali.pydict"


    self.gpus = []
    self.batch_dim = 0
    self.optimizer_name = "Adam"
    self.lr = 5e-2
    self.epoch_num   = 200
    self.batch_size_one_gpu  = 32
    self.batch_size_inference_one_gpu = 32
    self.dropout = 0.5
    self.eval_gap_instance_num = 5120

    self.use_polynormial_decay = True
    self.incremental_train = False