import mxnet as mx
import numpy as np
import time
import os
import logging
from datetime import datetime
from subprocess import call
from types import ModuleType

def setGPU(gpus):
    len_gpus = len(gpus.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    gpus = ','.join(map(str, range(len_gpus)))
    return gpus

def getTime():
    return datetime.now().strftime('%m-%d %H:%M:%S')

class Timer(object):
    curr_record = None
    prev_record = None

    @classmethod
    def record(cls):
        cls.prev_record = cls.curr_record
        cls.curr_record = time.time()

    @classmethod
    def interval(cls):
        if cls.prev_record is None:
            return 0
        return cls.curr_record - cls.prev_record

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
    if isinstance(args, ModuleType):
        args = vars(args)
    keys = [key for key in args.keys() if key[:2] != '__']
    keys.sort()
    length = max([len(x) for x in keys])
    msg = [('{:<'+str(length)+'}: {}').format(k, args[k]) for k in keys]

    msg = '\n' + '\n'.join(msg)
    info(logger, msg, color)

def loadParams(filename):
    data = mx.nd.load(filename)
    arg_params, aux_params = {}, {}
    for name, value in data.items():
        if name[:3] == 'arg':
            arg_params[name[4:]] = value
        elif name[:3] == 'aux':
            aux_params[name[4:]] = value
    if len(arg_params) == 0:
        arg_params = None
    if len(aux_params) == 0:
        aux_params = None
    return arg_params, aux_params

class SaveParams(object):
    def __init__(self, model, snapshot, model_name, num_save=5):
        self.model = model
        self.snapshot = snapshot
        self.model_name = model_name
        self.num_save = num_save
        self.save_params = []

    def save(self, n_epoch):
        self.save_params += [
                os.path.join(self.snapshot, '{}-{:04d}.params'.format(self.model_name, n_epoch)),
                os.path.join(self.snapshot, '{}-{:04d}.states'.format(self.model_name, n_epoch))]
        if isinstance(self.model, (list, tuple)):
            params_dict = {}
            for mod in self.model:
                arg_params, aux_params = mod.get_params()
                params_dict.update({'arg:'+k : v for k, v in arg_params.items()})
                params_dict.update({'aux:'+k : v for k, v in aux_params.items()})
            mx.nd.save(self.save_params[-2], params_dict)
        else:
            self.model.save_params(self.save_params[-2])
            self.model.save_optimizer_states(self.save_params[-1])

        if len(self.save_params) > 2 * self.num_save:
            call(['rm', self.save_params[0], self.save_params[1]])
            self.save_params = self.save_params[2:]
        return self.save_params[-2:]

    def __call__(self, n_epoch):
        return self.save(n_epoch)

def getLogger(snapshot, model_name):
    if not os.path.exists(snapshot):
        os.makedirs(snapshot)
    logging.basicConfig(filename=os.path.join(snapshot, model_name+'.log'), level=logging.INFO)
    logger = logging.getLogger()
    return logger

class LrScheduler(object):
    def __init__(self, method, init_lr, kwargs):
        self.method = method
        self.init_lr = init_lr

        if method == 'step':
            self.step_list = kwargs['step_list']
            self.factor = kwargs['factor']
            self.get = self._step
        elif method == 'poly':
            self.num_epoch = kwargs['num_epoch']
            self.power = kwargs['power']
            self.get = self._poly
        elif method == 'ramp':
            self.ramp_up = kwargs['ramp_up']
            self.ramp_down = kwargs['ramp_down']
            self.num_epoch = kwargs['num_epoch']
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
        lr = self.init_lr * ((1. - float(current_epoch)/self.num_epoch)**self.power)
        return lr

    def _ramp(self, current_epoch):
        if current_epoch < self.ramp_up:
            decay = np.exp(-(1 - float(current_epoch)/self.ramp_up)**2 * self.scale)
        elif current_epoch > (self.num_epoch - self.ramp_down):
            decay = np.exp(-(float(current_epoch+self.ramp_down-self.num_epoch)/self.ramp_down)**2 * self.scale)
        else:
            decay = 1.
        lr = self.init_lr * decay
        return lr

class GradBuffer(object):
    def __init__(self, model):
        self.model = model
        self.cache = None

    def write(self):
        if self.cache is None:
            self.cache = [[None if g is None else g.copyto(g.context) for g in g_list]\
                           for g_list in self.model._exec_group.grad_arrays]
        else:
            for gs_src, gs_dst in zip(self.model._exec_group.grad_arrays, self.cache):
                for g_src, g_dst in zip(gs_src, gs_dst):
                    if g_src is None:
                        continue
                    g_src.copyto(g_dst)

    def read_add(self):
        assert self.cache is not None
        for gs_src, gs_dst in zip(self.model._exec_group.grad_arrays, self.cache):
            for g_src, g_dst in zip(gs_src, gs_dst):
                if g_src is None:
                    continue
                g_src += g_dst

def initNormal(mean, std, name, shape):
    if name.endswith('_weight'):
        return mx.nd.normal(mean, std, shape)
    if name.endswith('_bias'):
        return mx.nd.zeros(shape)
    if name.endswith('_gamma'):
        return mx.nd.ones(shape)
    if name.endswith('_beta'):
        return mx.nd.zeros(shape)
    if name.endswith('_moving_mean'):
        return mx.nd.zeros(shape)
    if name.endswith('_moving_var'):
        return mx.nd.ones(shape)
    raise ValueError("Unknown name type for `{}`".format(name))

def checkParams(mod, arg_params, aux_params, auto_fix=True, initializer=mx.init.Normal(0.01), logger=None):
    arg_params = {} if arg_params is None else arg_params
    aux_params = {} if aux_params is None else aux_params

    arg_shapes = {name: array[0].shape for name, array in \
            zip(mod._exec_group.param_names, mod._exec_group.param_arrays)}
    aux_shapes = {name: array[0].shape for name, array in \
            zip(mod._exec_group.aux_names, mod._exec_group.aux_arrays)}

    extra_arg_params, extra_aux_params = [], []
    for name in arg_params.keys():
        if name not in arg_shapes:
            extra_arg_params.append(name)
    for name in aux_params.keys():
        if name not in aux_shapes:
            extra_aux_params.append(name)

    miss_arg_params, miss_aux_params = [], []
    for name in arg_shapes.keys():
        if name not in arg_params:
            miss_arg_params.append(name)
    for name in aux_shapes.keys():
        if name not in aux_params:
            miss_aux_params.append(name)

    mismatch_arg_params, mismatch_aux_params = [], []
    for name in arg_params.keys():
        if (name in arg_shapes) and (arg_shapes[name] != arg_params[name].shape):
            mismatch_arg_params.append(name)
    for name in aux_params.keys():
        if (name in aux_shapes) and (aux_shapes[name] != aux_params[name].shape):
            mismatch_aux_params.append(name)
    
    for name in extra_arg_params:
        info(logger, "Find extra arg_params: {}: given {}".format(name, arg_params[name].shape), 'red')
    for name in extra_aux_params:
        info(logger, "Find extra aux_params: {}: given {}".format(name, aux_params[name].shape), 'red')
    for name in miss_arg_params:
        info(logger, "Find missing arg_params: {}: target {}".format(name, arg_shapes[name]), 'red')
    for name in miss_aux_params:
        info(logger, "Find missing aux_params: {}: target {}".format(name, aux_shapes[name]), 'red')
    for name in mismatch_arg_params:
        info(logger, "Find mismatch arg_params: {}: given {}, target {}".format(
            name, arg_params[name].shape, arg_shapes[name]), 'red')
    for name in mismatch_aux_params:
        info(logger, "Find mismatch aux_params: {}: given {}, target {}".format(
            name, aux_params[name].shape, aux_shapes[name]), 'red')

    if len(extra_arg_params + extra_aux_params + \
           miss_arg_params + miss_aux_params + \
           mismatch_arg_params + mismatch_aux_params) == 0:
        return arg_params, aux_params

    if not auto_fix:
        info(logger, "Bad params not fixed.", 'red')
        return arg_params, aux_params

    for name in (extra_arg_params + mismatch_arg_params):
        del arg_params[name]
    for name in (extra_aux_params + mismatch_aux_params):
        del aux_params[name]

    attrs = mod._symbol.attr_dict()
    for name in (miss_arg_params + mismatch_arg_params):
        arg_params[name] = mx.nd.zeros(arg_shapes[name])
        try:
            initializer(mx.init.InitDesc(name, attrs.get(name, None)), arg_params[name])
        except ValueError:
            initializer(name, arg_params[name])
    for name in (miss_aux_params + mismatch_aux_params):
        aux_params[name] = mx.nd.zeros(aux_shapes[name])
        try:
            initializer(mx.init.InitDesc(name, attrs.get(name, None)), aux_params[name])
        except ValueError:
            initializer(name, aux_params[name])
    info(logger, "Bad params auto fixed successfully.", 'red')
    return arg_params, aux_params

class EMAShareModel(object):
    def __init__(self, mod, momentum):
        self.momentum = momentum
        self.mod = mod
        self.arg_params = None
        self.aux_params = None
        self.init_params()

    def init_params(self, arg_params=None, aux_params=None):
        arg_params_, aux_params_ = self.mod.get_params()
        self.arg_params = arg_params_ if arg_params is None else arg_params
        self.aux_params = aux_params_ if aux_params is None else aux_params

    def update_params(self, arg_params, aux_params):
        for k, v in self.arg_params.items():
            (v * self.momentum + arg_params[k] * (1 - self.momentum)).copyto(v)
        for k, v in self.aux_params.items():
            (v * self.momentum + aux_params[k] * (1 - self.momentum)).copyto(v)

    def forward_and_update(self, *args, **kwargs):
        curr_args, curr_auxs = self.mod.get_params()
        self.mod.set_params(arg_params=self.arg_params, aux_params=self.aux_params)
        self.mod.forward(*args, **kwargs)
        outputs = [x.copyto(mx.cpu()) for x in self.mod.get_outputs()]
        self.mod.set_params(arg_params=curr_args, aux_params=curr_auxs)
        self.update_params(curr_args, curr_auxs)
        return outputs

class EMAModel(object):
    def __init__(self, mod, momentum):
        self.momentum = momentum
        self._mod = mod
        self._mod_ema = self.create_eam_model()
        self.init_kvstore()

    def create_eam_model(self):
        mod = mx.mod.Module(
                symbol = self._mod._symbol,
                data_names = self._mod._data_names,
                label_names = self._mod._label_names,
                context = self._mod._context,
                fixed_param_names = self._mod._fixed_param_names
        )
        mod.bind(
                data_shapes = self._mod._data_shapes,
                label_shapes = self._mod._label_shapes
        )

        arg_params, aux_params = self._mod.get_params()
        mod.init_params(arg_params=arg_params, aux_params=aux_params)
        return mod

    def init_kvstore(self):
        self.kvstore = mx.kv.create('device')
        self.kvstore._set_updater(EMAModel.ema_updater(self.momentum))

        assert self._mod_ema._exec_group.param_names == self._mod._exec_group.param_names
        assert self._mod_ema._exec_group.aux_names == self._mod._exec_group.aux_names

        self.arg_names = ['arg:' + name for name in self._mod_ema._exec_group.param_names]
        self.aux_names = ['aux:' + name for name in self._mod_ema._exec_group.aux_names]
        for k, v in zip(self.arg_names, self._mod_ema._exec_group.param_arrays):
            self.kvstore.init(k, v[0])
        for k, v in zip(self.aux_names, self._mod_ema._exec_group.aux_arrays):
            self.kvstore.init(k, v[0])

    @staticmethod
    def ema_updater(momentum):
        def wrapper(key, inputs, stored):
            (stored*momentum + inputs.as_in_context(stored.context)*(1 - momentum)).copyto(stored)
        return wrapper

    def update(self):
        for i, (k, v, v_ema) in enumerate(zip(self.arg_names, self._mod._exec_group.param_arrays, self._mod_ema._exec_group.param_arrays)):
            self.kvstore.push(k, v[0], priority=-i)
            self.kvstore.pull(k, v_ema, priority=-i)
        for i, (k, v, v_ema) in enumerate(zip(self.aux_names, self._mod._exec_group.aux_arrays, self._mod_ema._exec_group.aux_arrays)):
            self.kvstore.push(k, v[0], priority=-i)
            self.kvstore.pull(k, v_ema, priority=-i)

    def collect_grad(self):
        for grad, grad_ema in zip(self._mod._exec_group.grad_arrays, self._mod_ema._exec_group.grad_arrays):
            for _grad, _grad_ema in zip(grad, grad_ema):
                _grad += grad_ema * (1 - self.momentum)

    def forward(self, *args, **kwargs):
        self._mod_ema.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        self._mod_ema.backward(*args, **kwargs)

    def get_outputs(self, *args, **kwargs):
        return self._mod_ema.get_outputs(*args, **kwargs)
    
