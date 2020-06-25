
from pa_nlp.pytorch.estimator.param import ParamBase

class Param(ParamBase):
  def __init__(self):
    super(Param, self).__init__("ver_1_classification")

    self.train_files = 'results/v1/exp_7/ivectors_train.7'
    self.vali_file = 'results/v1/exp_7/ivectors_test.7'
    self.class_num = 3
    self.gpus = []
    self.batch_dim = 0
    self.optimizer_name = "Adam"
    self.lr = 5e-5
    self.epoch_num   = 20
    self.batch_size_one_gpu  = 32
    self.batch_size_inference_one_gpu = 32

    self.eval_gap_instance_num = 1000
    self.train_sample_num = 180000
    self.use_polynormial_decay = True
    self.incremental_train = False
    self.label_smoothing = 0.05