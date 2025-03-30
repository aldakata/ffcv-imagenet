import torch as ch
from torch.amp import GradScaler
from torch.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from datetime import timedelta

import wandb

from torchvision import models
import torchmetrics
import numpy as np
from tqdm import tqdm

import os
import time
import json
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from functools import partial


from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.__dir__())), default='resnet18'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True),
    num_classes=Param(int, 'num classes', default=1000),
    processing=Param(int, 'class of processing, 0: none, 1: no duplicates', default=0),

)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic', 'eye']), default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
)

Section('loss', 'loss function ablation').params(
    loss=Param(OneOf(['ova', 'sigmoid', 'ce']), 'type of objective function', default='ce'),
    mu=Param(float, 'weight of the negative heads class logits', default=0),
    lmbd=Param(float, 'weight of the negative heads sigmoid loss', default=0),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=1),
    experiment_name=Param(str, 'name of the experiment', required=True)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd', 'adam'])), 'The optimizer', default='sgd'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    negative_label_smoothing=Param(float, 'label smoothing parameter', default=0),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0),
    negative_update=Param(int, 'update with neg loss?', default=1)
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0
    num_steps = epoch // step_length
    return step_ratio**num_steps * lr
@param('lr.lr')
def get_eye_lr(epoch, lr):
    return lr

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]

class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)

class SigmoidLoss(ch.nn.Module):
    def __init__(self, mu, lmbd, num_classes, negative_label_smoothing):
        super(SigmoidLoss, self).__init__()
        self.num_classes = num_classes
        self.mu, self.lmbd = mu, lmbd
        self.negative_label_smoothing = negative_label_smoothing

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
        multiclass_bce_targets = hot1*charges[:, None] # 1s for positive samples at positive class, 0 elsewhere
        multiclass_bce_targets[charges==0] += (1-hot1[charges==0])*self.negative_label_smoothing # for the negative samples, 0 for the selected class, negative_label_smoothing for the rest

        l = ch.nn.functional.binary_cross_entropy_with_logits(logits, multiclass_bce_targets, weight=weights, reduction='none')
        l = l/bs
        return l

@param('loss.loss')
@param('loss.mu')
@param('loss.lmbd')
@param('training.negative_label_smoothing')
def get_loss_fn(loss, mu, lmbd, label_smoothing, num_classes, negative_label_smoothing):
    if loss == 'sigmoid':
        return SigmoidLoss(mu, lmbd, num_classes, negative_label_smoothing)
    elif loss == 'ce':
        return lambda x,y,z=None: ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)(x,y) # Hacky but works
    else:
        raise ValueError(f'Unknown loss {loss}')

class ImageNetTrainer:
    @param('training.distributed')
    @param('logging.experiment_name')
    def __init__(self, gpu, distributed, experiment_name):
        self.restart_count = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", 0))
        self.snapshot_path = f'/mnt/lustre/work/oh/owl156/tmp/snapshot_{experiment_name}.pt'      
        self.all_params = get_current_config()
        self.gpu = gpu

        self.get_num_classes()

        if distributed:
            self.setup_distributed()
        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model = self.create_model()
        self.maybe_load_snapshot()
        
        self.model = self.model.to(memory_format=ch.channels_last)
        self.model = self.model.to(self.gpu)
        if distributed:
            self.model = ch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu])

        self.scaler = GradScaler()
        self.create_optimizer()
        self.initialize_logger()

    def _save_snapshot(self, epoch):
        if self.gpu == 0 and False:
            snapshot = {}
            snapshot["MODEL_STATE"] = self.model.module.state_dict()
            snapshot["EPOCHS_RUN"] = epoch
            snapshot["uid"] = self.uid
            ch.save(snapshot, f"{self.snapshot_path}")
            print(f"Epoch {epoch} | Training snapshot saved at{self.snapshot_path}")

    def maybe_load_snapshot(self):
        self.should_load() # check if first restart and prapare for it
        # Load
        if os.path.exists(self.snapshot_path) and self.restart_count > 0 and False:
            snapshot = ch.load(self.snapshot_path)
            self.model.load_state_dict(snapshot["MODEL_STATE"])
            self.epochs_run = snapshot["EPOCHS_RUN"]
            self.uid = snapshot["uid"]
            print(f"Resuming training from snapshot at Epoch {self.epochs_run} restart count {self.restart_count}")

        # Normal Init
        else:
            self.uid = str(uuid4())
            self.epochs_run = 0

    def should_load(self):
        self.restart_count = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", 0))
        # Make sure that there is no snapshot from another failed run
        if self.restart_count == 0:
            self.clean_snapshot()

    def clean_snapshot(self):
        if os.path.exists(self.snapshot_path) and self.gpu == 0:
            os.remove(self.snapshot_path)

    @param('data.num_classes')
    def get_num_classes(self, num_classes):
        self.num_classes=num_classes

    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        # os.environ['MASTER_ADDR'] = address
        # os.environ['MASTER_PORT'] = port
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=1200))
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr,
            'eye': get_eye_lr,
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    @param('lr.lr')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing, lr):
        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]
        if optimizer == 'sgd':
            self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        else:
            self.optimizer = ch.optim.Adam(param_groups, lr=lr)
        self.loss = get_loss_fn(label_smoothing=label_smoothing, num_classes=self.num_classes)

    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
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
        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines=pipelines,
                        distributed=distributed)
        return loader

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    @param('training.epochs')
    @param('logging.log_level') 
    def train(self, epochs, log_level):
        for epoch in range(self.epochs_run, epochs):
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)
            train_loss, loss_p, loss_n = self.train_loop(epoch)

            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss,
                    'loss_p': loss_p,
                    'loss_n': loss_n,
                    'epoch': epoch
                }
                if epoch % 5==0:
                    self._save_snapshot(epoch)
                    self.eval_and_log(extra_dict)
        self.eval_and_log({'epoch':epoch})
        if self.gpu == 0:
            ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')
        # remove tmp checkpoint
        self.clean_snapshot()

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            data = dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'top_1': stats['top_1'],
                'top_5': stats.get('top_5',0),
                'val_loss': stats['loss'],
                'val_time': val_time
            }, **extra_dict)
            self.log(data)
            wandb.log(data)
        return stats

    @param('model.arch')
    @param('model.pretrained')
    @param('training.distributed')
    @param('training.use_blurpool')
    def create_model(self, arch, pretrained, distributed, use_blurpool):
        model = getattr(models, arch)(pretrained=pretrained,  num_classes=self.num_classes)
        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)
        return model

    @param('logging.log_level')
    @param('training.negative_update')
    def train_loop(self, epoch, log_level, negative_update):
        model = self.model
        model.train()
        losses = []
        losses_p = []
        losses_n = []

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target, charges, _) in enumerate(iterator):

            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                output = self.model(images)
                loss_train = self.loss(output, target, charges)
                loss_pos = loss_train[charges==1].sum()
                loss_neg = loss_train[charges==0].sum()
                loss_train = loss_pos+negative_update*loss_neg
            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            ### Training end

            ### Logging start
            if log_level > 0:
                losses.append(loss_train.detach())
                losses_p.append(loss_pos.detach())
                losses_n.append(loss_neg.detach())
                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ['loss', 'loss_p','loss_n']
                    values += [f'{loss_train.item():.3f}', f'{loss_pos.item():.3f}', f'{loss_neg.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)
        return ch.stack(losses).mean().item(), ch.stack(losses_p).mean().item(), ch.stack(losses_n).mean().item()
            ### Logging end

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()
        if self.num_classes>5:
            topk = ['top_1', 'top_5']
        else:
            topk = ['top_1']
        with ch.no_grad():
            with autocast('cuda'):
                for images, target in tqdm(self.val_loader):
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    for k in topk:
                        self.val_meters[k](output, target)

                    loss_val = self.loss(output, target)
                    self.val_meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('logging.folder')
    @param('logging.experiment_name')
    def initialize_logger(self, folder, experiment_name):
        if self.num_classes>5:
            tmp = {            
                   'top_5': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, top_k=5).to(self.gpu)
                   }
        else:
            tmp={}
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes).to(self.gpu),
            **tmp,
            'loss': MeanScalarMetric().to(self.gpu)
        }

        if self.gpu == 0:
            folder = (Path(folder) / str(self.uid)).absolute()
            folder.mkdir(parents=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)

            negative_dataset_type = self.all_params["data.train_dataset"].split('/')[-1].split('_')[0]
            lr = self.all_params['lr.lr']
            mu = self.all_params['loss.mu']
            lmbd = self.all_params['loss.lmbd']
            neg_counts = self.all_params["data.train_dataset"].split('_')[-1].split('.')[0]
            num_classes = ""
            if self.all_params["data.num_classes"] == 100:
                num_classes = "_100"
            elif self.all_params["data.num_classes"] == 4:
                num_classes = "_4"
            else:
                num_classes = "_" + str(self.all_params["data.num_classes"])
            name = f'{negative_dataset_type}_{lr}_{mu}_{lmbd}_{neg_counts}_{experiment_name}{num_classes}'
            """Initalizes a wandb run"""
            os.environ["WANDB__SERVICE_WAIT"] = "600"
            os.environ["WANDB_SILENT"] = "true"

            wandb.init(
                project='NegativeSamples1K',
                dir='wandb/',
                config=params,
                name=name,
                id=self.uid,
                resume="allow",
            )

    def log(self, content):
        print(f'=> Log: {content}')
        if self.gpu != 0: return
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            # fd.flush() # albert ? 

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    def launch_from_args(cls, distributed, world_size):
        make_config(quiet=True)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        cls.exec(local_rank)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()

# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    
    if not quiet:
        config.summary()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    make_config()
    ImageNetTrainer.launch_from_args()
