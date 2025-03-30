import torch as ch
import numpy as np
import json
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from pathlib import Path
from typing import List
from torchvision import transforms, datasets, models
import torch.nn.functional as F
from classes import IMAGENET2012_CLASSES
keys = IMAGENET2012_CLASSES.keys()
values = IMAGENET2012_CLASSES.values()
indeces = list(range(1000))
ntof = dict(zip(values, keys))
fton = IMAGENET2012_CLASSES

itof = dict(zip(indeces, keys))
ftoi = dict(zip(keys, indeces))

ntoi = dict(zip(values, indeces))
iton = dict(zip(indeces, values))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

def create_train_loader(train_dataset, num_workers, batch_size,
                        distributed, in_memory,this_device):
    train_path = Path(train_dataset)
    assert train_path.is_file()

    decoder = RandomResizedCropRGBImageDecoder((224, 224))
    image_pipeline: List[Operation] = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(ch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]
    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(ch.device(this_device), non_blocking=True),
    ]
    charge_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(ch.device(this_device), non_blocking=True),
    ]
    index_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(ch.device(this_device), non_blocking=True),
    ]
    pipelines = {
        "image": image_pipeline,
        "label": label_pipeline,
        "charge": charge_pipeline,
    }
    order = OrderOption.RANDOM
    loader = Loader(train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=True,
                    pipelines=pipelines,
                    distributed=distributed)
    return loader
class SigmoidLoss(ch.nn.Module):
    def __init__(self, mu, lmbd, num_classes):
        super(SigmoidLoss, self).__init__()
        self.num_classes = num_classes
        self.mu, self.lmbd = mu, lmbd

    def forward(self, logits, targets, charges=None):
        if charges is None:
            hot1 = F.one_hot(targets, num_classes=self.num_classes).float()
            l = ch.nn.functional.binary_cross_entropy_with_logits(logits, hot1, reduction='sum')
            bs = logits.size(0)
            l = l/bs # this way you only reduce over the batch size and not over the number of classes and batch size
            return l
        hot1 = F.one_hot(targets, num_classes=self.num_classes).float()
        bs = logits.size(0)
        weights = ch.ones_like(logits)
        weights[charges == 0] = self.lmbd
        weights[charges == 0] += (self.mu-self.lmbd)*hot1[charges == 0]
        l = ch.nn.functional.binary_cross_entropy_with_logits(logits, hot1*charges[:, None], weight=weights, reduction='none')
        l = l/bs
        return l
num_classes = 4
mu, lmbd = 1, 0.1

device=ch.device('cuda')
model = models.resnet50(num_classes=num_classes)
model = model.to(device)
p = '/mnt/lustre/work/oh/owl156/data/dataset_negative_4_GCP_400_0.5_90_150.beton'
loss = SigmoidLoss(mu, lmbd, num_classes)

from torch.amp import autocast
loader = create_train_loader(p, 8, 4, 0, 1, device)
for i, (img,lbl,chg, _) in enumerate(loader):
    img,lbl,chg=img.to(device),lbl.to(device),chg.to(device)
    with autocast('cuda'):
        logits = model(img)
    if any(chg==0):
        loss_train = loss(logits, lbl, chg)
        loss_pos = loss_train[chg==1].sum()
        loss_neg = loss_train[chg==0].sum()
        lt = loss_pos+loss_neg
        break

logits, targets, charges = logits,lbl,chg
nls = 0.5
hot1 = F.one_hot(targets, num_classes=num_classes).float()
bs = logits.size(0)
weights = ch.ones_like(logits)
weights[charges == 0] = lmbd
weights[charges == 0] += (mu-lmbd)*hot1[charges == 0]
multiclass_bce_targets = hot1*charges[:, None]
print(lbl, charges)
multiclass_bce_targets[charges==0] += (1-hot1[charges==0])*nls
print(multiclass_bce_targets, '\n', hot1)
l = ch.nn.functional.binary_cross_entropy_with_logits(logits, multiclass_bce_targets, weight=weights, reduction='none')
l = l/bs