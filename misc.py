import torch
#torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import DataLoader

from core.data.VOC import VOCVersatileDataset
from core.data.COCO import COCOVersatileDataset
from core.utils import *

from compute_iou import *
import pydensecrf.densecrf as dcrf

import os


def config_env(gpu, args, port):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.cuda.set_device(gpu)
    world_size = len(args.gpus.split(','))
    is_distributed = world_size > 1
    if is_distributed:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = port
        dist.init_process_group('nccl', rank=gpu, world_size=world_size)
    return is_distributed, world_size

def get_snapshot_name(args):
    try:
        assert args.model_name is not None
        dirname = f'{args.model_name}_T{args.threshold}_C{args.capacity}'
    except:
        dirname = f'{args.model}_T{args.threshold}_C{args.capacity}'
    if args.suffix:
        dirname += f'_{args.suffix}'

    return dirname

def get_loader(args, subset, augmentation=False, segment_root=None, return_src=False, downsample_label=1):
    if args.dataset == 'voc':
        assert subset in ['train_aug', 'val', 'test', 'train_aug_val'], subset
    elif args.dataset == 'coco':
        pass
    else:
        raise RuntimeError(f"No subset '{subset}' for dataset '{dataset}'")
    
    world_size = len(args.gpus.split(','))
    is_distributed = world_size > 1
    size = [int(x) for x in args.train_size.split(',')] if augmentation else None

    DatasetSome = eval(f'{args.dataset.upper()}VersatileDataset')
    image_root = args.image_root if 'train' in subset else args.image_val_root
    ann_root = None if 'test' in subset else (args.ann_root if 'train' in subset else args.ann_val_root)
    dataset = DatasetSome(image_root, subset, size,
            segment_root=segment_root, superpixel_root=None,
            annotation_root=ann_root,
            image_label_use_background=True,
            return_src=return_src,
            rand_crop=augmentation, rand_mirror=augmentation, rand_scale=(0.5, 1.5) if augmentation else None,
            downsample_label=downsample_label
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=augmentation) if is_distributed else None
    device_bs = (args.batch_size // world_size) if augmentation else 1
    num_workers = ( (args.num_workers + world_size - 1) // world_size ) if augmentation else 1
    loader = DataLoader(dataset, batch_size=device_bs, shuffle=augmentation and (sampler is None),
            pin_memory=False, drop_last=augmentation, sampler=sampler, num_workers=num_workers)
    return loader, sampler

def freeze_bn(mod):
    num_bn = 0
    for m in mod.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            num_bn += 1
    if num_bn > 0:
        print(f'Freeze {num_bn} BN layers.')

def get_color_cam(image, cam, palette):
    cam = torch.clamp_min(cam, 0) / torch.clamp_min(cam.max(1, keepdim=True)[0].max(2, keepdim=True)[0], 1e-5)
    intensity, index = cam.max(0)
    colormap = palette[(index + 1).data.cpu().numpy().ravel()].reshape(index.shape+(3,))
    intensity = intensity.data.cpu().numpy()**0.3
    hsv = cv2.cvtColor(colormap, cv2.COLOR_BGR2HSV)
    hsv[..., -1] = (intensity * 255.).astype(np.uint8)
    colormap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    h, w = image.shape[:2]
    colormap = cv2.resize(colormap, (w, h))
    colormap = cv2.addWeighted(colormap, 0.85, image, 0.15, 0)
    return colormap

def resize_maybe(src, tgt, mode='bilinear'):
    if isinstance(tgt, torch.Tensor):
        h, w = tgt.shape[-2:]
    else:
        h, w = tgt

    if (src.size(2) != h) or (src.size(3) != w):
        src = F.interpolate(src, (h, w), mode=mode, align_corners=None if mode == 'nearest' else True)
    return src

def crf_postprocessing(src, probs, save, niter=5):
    img = cv2.imread(src)[..., ::-1].copy()
    h, w = img.shape[:2]

    if isinstance(probs, str):
        assert os.path.exists(probs), probs
        probs = np.load(probs)
    elif isinstance(probs, torch.Tensor):
        probs = probs.data.cpu().numpy()

    probs = probs.astype(np.float32)
    assert list(probs.shape[1:]) == [h, w], probs.shape

    d = dcrf.DenseCRF2D(w, h, probs.shape[0])
    u = - probs.reshape(probs.shape[0], -1)
    d.setUnaryEnergy(u)

    d.addPairwiseGaussian(sxy=2, compat=2)
    d.addPairwiseBilateral(sxy=65, srgb=3, rgbim=img, compat=4)

    probs_crf = d.inference(niter)
    probs_crf = np.array(probs_crf).reshape(-1, h, w)
    pred_crf = probs_crf.argmax(0).astype(np.uint8)

    if save is None:
        return pred_crf
    imwrite(save, pred_crf)

class Scheduler(object):
    def __init__(self):
        self.cnt = 0

    def next(self):
        val = self.get(self.cnt)
        self.cnt += 1
        return val

class RampUpScheduler(Scheduler):
    def __init__(self, rampup, scale):
        super(RampUpScheduler, self).__init__()
        self.rampup = rampup
        self.scale = scale
        self.params = [-2./float(rampup)**3, 3./float(rampup)**2]

    def get(self, x):
        if x < self.rampup:
            return (self.params[0] * float(x)**3 + self.params[1] * float(x)**2)**self.scale
        return 1.0

class LinearScheduler(Scheduler):
    def __init__(self, begin, end, steps):
        super(LinearScheduler, self).__init__()
        self.steps = steps
        self.begin = begin
        self.end = end
        self.div = (end - begin) / steps

    def get(self, x):
        val = self.end if x > self.steps else self.begin
        self.begin += self.div
        return val
