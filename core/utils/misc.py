import numpy as np
import time
import os
from datetime import datetime
from subprocess import call
from types import ModuleType
from collections import OrderedDict
import torch
import torch.utils.tensorboard

class Logger(object):
    def __init__(self, filename):
        self.filename = filename

    def info(self, msg):
        with open(self.filename, 'a') as f:
            f.write(str(msg) + '\n')

# def getLogger(snapshot, model_name):
#     import logging
#     if not os.path.exists(snapshot):
#         os.makedirs(snapshot)
#     logging.basicConfig(filename=os.path.join(snapshot, model_name+'.log'), level=logging.INFO)
#     logger = logging.getLogger("some logger")
#     return logger

def getLogger(snapshot, model_name):
    if not os.path.exists(snapshot):
        os.makedirs(snapshot)
    logger = Logger(filename=os.path.join(snapshot, model_name+'.log'))
    print(logger, logger.filename)
    return logger

class IoUMeter(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confmat = None
        self.reset()

    def reset(self):
        self.confmat = np.zeros((self.num_classes, self.num_classes), np.float32)

    def put(self, gt, pred):
        assert gt.shape == pred.shape
        index = (gt < self.num_classes) & (pred < self.num_classes)
        self.confmat += np.bincount(gt[index].astype(np.int64) * self.num_classes + pred[index].astype(np.int64),
                minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def get(self):
        return np.diag(self.confmat) / np.maximum(self.confmat.sum(axis=0) + self.confmat.sum(axis=1) - np.diag(self.confmat), 1)

    def item(self):
        return float(self.get().mean())

    def __str__(self):
        return '{:.4f}'.format(self.item())

def setGPU(gpus):
    len_gpus = len(gpus.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    gpus = ','.join(map(str, range(len_gpus)))
    return gpus

def getTime():
    return datetime.now().strftime('%m-%d %H:%M:%S')

class Timer(object):
    def __init__(self):
        self.curr_record = time.time()
        self.prev_record = time.time()

    def record(self):
        self.prev_record = self.curr_record
        self.curr_record = time.time()

    def interval(self):
        if self.prev_record is None:
            return 0
        return self.curr_record - self.prev_record

class TensorBoardWriter(object):
    def __init__(self, filename):
        self.writer = torch.utils.tensorboard.SummaryWriter(filename)
        self.global_steps = OrderedDict()

    def add_scalar(self, k, v):
        step = self.global_steps.get(k, 0)
        self.writer.add_scalar(k, v, step)
        self.global_steps[k] = step + 1
    
    def add_scalars(self, kv_dict):
        for k, v in kv_dict.items():
            self.add_scalar(k, v)
    
    def add_image(self, k, v):
        step = self.global_steps.get(k, 0)
        self.writer.add_image(k, v[..., ::-1], step, dataformats='HWC')
        self.global_steps[k] = step + 1

    def flush(self):
        self.writer.flush()
    
    def close(self):
        self.writer.close()

class AvgMeter(object):
    def __init__(self, *args, auto_init=True):
        self.data = OrderedDict()
        for k in args:
            self.data[k] = []
        self._auto_init = auto_init
        self.tb_writer = None

    def clean(self):
        for k in self.data.keys():
            self.data[k] = []

    def init(self, *args):
        for k in args:
            self.data[k] = []

    def put(self, k, v):
        if k not in self.data:
            if self._auto_init:
                self.init(k)
            else:
                raise KeyError("Key '{}' not init yet.".format(k))
        self.data[k].append(v)

        if self.tb_writer is not None:
            self.tb_writer.add_scalar(k, v)

    def get(self, k):
        values = self.data[k]
        return sum(values) / len(values)

    def __str__(self):
        res = []
        for k, v_list in self.data.items():
            v = sum(v_list) / max(len(v_list), 1)
            res.append('{}={:.4f}'.format(k, v))
        return ', '.join(res)
    
    def bind_tb_writer(self, tb_writer):
        assert isinstance(tb_writer, TensorBoardWriter), type(tb_writer)
        self.tb_writer = tb_writer

def wrapColor(string, color):
    try:
        header = {
                'red':       '\033[91m',
                'green':     '\033[92m',
                'yellow':    '\033[93m',
                'blue':      '\033[94m',
                'purple':    '\033[95m',
                'cyan':      '\033[96m',
                'darkcyan':  '\033[36m',
                'bold':      '\033[1m',
                'underline': '\033[4m'}[color.lower()]
    except KeyError:
        raise ValueError("Unknown color: {}".format(color))
    return header + string + '\033[0m'

def info(logger, msg, color=None):
    msg = '[{}]'.format(getTime()) + msg
    if logger is not None:
        logger.info(msg)

    if color is not None:
        msg = wrapColor(msg, color)
    print(msg)

def summaryArgs(logger, args, color=None):
    args = vars(args)
    keys = [key for key in args.keys() if key[:2] != '__']
    keys.sort()
    length = max([len(x) for x in keys])
    msg = [('{:<'+str(length)+'}: {}').format(k, args[k]) for k in keys]

    msg = '\n' + '\n'.join(msg)
    info(logger, msg, color)

class SaveParams(object):
    def __init__(self, model, snapshot, model_name, num_save=5):
        self.model = model
        self.snapshot = snapshot
        self.model_name = model_name
        self.num_save = num_save
        self.cache_files = []

    def save(self, epoch):
        filename = os.path.join(self.snapshot, 'checkpoint', '{}-{:04d}.pth'.format(self.model_name, epoch))
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        torch.save(self.model.state_dict(), filename)
        self.cache_files.append(filename)

        while len(self.cache_files) > self.num_save:
            call(['rm', self.cache_files[0]])
            self.cache_files = self.cache_files[1:]

    def __call__(self, epoch):
        return self.save(epoch)

    @property
    def filename(self):
        return '' if len(self.cache_files) == 0 else self.cache_files[-1]

class LrScheduler(object):
    def __init__(self, method, init_lr, kwargs):
        self.method = method
        self.init_lr = init_lr

        if method == 'step':
            self.step_list = kwargs['step_list']
            self.factor = kwargs['factor']
            self.get = self._step
        elif method == 'poly':
            self.num_epochs = kwargs['num_epochs']
            self.power = kwargs['power']
            self.get = self._poly
        elif method == 'ramp':
            self.ramp_up = kwargs['ramp_up']
            self.ramp_down = kwargs['ramp_down']
            self.num_epochs = kwargs['num_epochs']
            self.scale = kwargs['scale']
            self.get = self._ramp
        else:
            raise ValueError(method)

    def _step(self, current_epoch):
        lr = self.init_lr
        step_list = [x for x in self.step_list]
        while len(step_list) > 0 and current_epoch >= step_list[0]:
            lr *= self.factor
            del step_list[0]
        return lr

    def _poly(self, current_epoch):
        lr = self.init_lr * ((1. - float(current_epoch)/self.num_epochs)**self.power)
        return lr

    def _ramp(self, current_epoch):
        if current_epoch < self.ramp_up:
            decay = np.exp(-(1 - float(current_epoch)/self.ramp_up)**2 * self.scale)
        elif current_epoch > (self.num_epochs - self.ramp_down):
            decay = np.exp(-(float(current_epoch+self.ramp_down-self.num_epochs)/self.ramp_down)**2 * self.scale)
        else:
            decay = 1.
        lr = self.init_lr * decay
        return lr

