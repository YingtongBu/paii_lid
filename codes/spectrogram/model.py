import fastai
from fastai.vision import *
print(fastai.__version__)

#import torch
#fastai.torch_core.defaults.device = torch.device('cpu')

imagepath = '/data/pytong/spectrogram/youtube'
#testpath = '/home/buyingtong/model2_images/test/'

# Data augmentation
tfms = get_transforms(do_flip=False, max_rotate=None)
data = (ImageList.from_folder(imagepath)
        .split_by_rand_pct(0.2) #split train/valid
        .label_from_folder() #label depending on the folder of the filenames
        #.add_test_folder(testpath) #Optionally add a test set
        .transform(size=(500,250))#transform(tfms, size=500)
        .databunch(bs=32)).normalize(imagenet_stats)
print(data.train_dl) #DeviceDataLoader
print(data.train_ds)


learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(3) #Only train FC layers
learn.unfreeze() #begin to train Conv layers
learn.fit_one_cycle(3)
learn.freeze() #freeze Conv layers


data_test = (ImageList.from_folder(imagepath)
             .split_by_folder(train='train', valid='lre07')
             .label_from_folder()
             .transform(tfms, size=500)
             .databunch()).normalize(imagenet_stats)
print(data_test.train_dl) #DeviceDataLoader
print(data_test.train_ds)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


learn = cnn_learner(data_test, models.resnet18, metrics=accuracy)
learn.unfreeze() #begin to train Conv layers
learn.fit_one_cycle(10)
learn.freeze() #freeze Conv layers
learn.fit_one_cycle(3) #Only train FC layers

learn.validate(data_test.valid_dl)