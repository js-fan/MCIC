from traineval_cam import *
from traineval_seg import *
from traineval_e2e import *
from traineval_mcis import *
from traineval_mcis_2stage import *
from misc import *

import argparse
import copy
import torch.multiprocessing as mp

def run(args, mode):
    args.seed = np.random.randint(0, 10000)
    world_size = len(args.gpus.split(','))
    run_script = eval('run_' + mode)
    args = copy.copy(args)

    if world_size > 1:
        port = str(np.random.randint(2048, 65536))
        mp.spawn(run_script, args=(args, port), nprocs=world_size, join=True)
    else:
        run_script(0, args, 0)

def _voc_default_path(subset=None):
    img = 'data/VOC2012/JPEGImages'
    gt = 'data/VOC2012/extra/SegmentationClassAug'
    sal='./seeds/saliency_aug'
    ann='data/VOC2012/Annotations'
    return img, gt, sal, ann

def _coco_default_path(subset):
    assert subset in ['train', 'val'], subset
    img = f'data/coco/{subset}2014'
    gt = f'coco/converted_labels/segmentation/{subset}2014'
    sal = f'coco/saliency/saliency_crf/{subset}2014'
    ann = f'coco/converted_labels/{subset}2014.txt'
    return img, gt, sal, ann

def _set_default_roots(args):
    img, gt, sal, ann = eval(f'_{args.dataset}_default_path')('train')
    args.image_root = img if args.image_root is None else args.image_root
    args.gt_root = gt if args.gt_root is None else args.gt_root
    args.seed_root = sal if args.seed_root is None else args.seed_root 
    args.ann_root = ann if args.ann_root is None else args.ann_root

    img, gt, sal, ann = eval(f'_{args.dataset}_default_path')('val')
    args.image_val_root = img if args.image_val_root is None else args.image_val_root
    args.gt_val_root = gt if args.gt_val_root is None else args.gt_val_root
    args.seed_val_root = sal if args.seed_val_root is None else args.seed_val_root 
    args.ann_val_root = ann if args.ann_val_root is None else args.ann_val_root
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', type=str, default=None)
    parser.add_argument('--gt-root', type=str, default=None)
    parser.add_argument('--seed-root', type=str, default=None)
    parser.add_argument('--ann-root', type=str, default=None)
    parser.add_argument('--image-val-root', type=str, default=None)
    parser.add_argument('--gt-val-root', type=str, default=None)
    parser.add_argument('--seed-val-root', type=str, default=None)
    parser.add_argument('--ann-val-root', type=str, default=None)
    parser.add_argument('--sp-root', type=str, default=None)

    parser.add_argument('--model', type=str, default='vgg16_largefov')
    parser.add_argument('--train-size', type=str, default='321,321')
    parser.add_argument('--test-scales', type=str, default='0.75,1,1.25')
    parser.add_argument('--num-classes', type=int, default=None)

    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sgd_mom', type=float, default=0.9)
    parser.add_argument('--sgd_wd', type=float, default=5e-4)

    parser.add_argument('--force-snapshot', type=str, default=None)
    parser.add_argument('--snapshot', type=str, default='../Snapshots/MultiImage')
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--log-frequency', type=int, default=25)
    parser.add_argument('--pretrained', type=str, default='./pretrained/vgg16_aspp.pth')

    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--non-strict', action='store_true')

    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--capacity', type=int, default=64)

    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--ema-momentum', type=float, default=0.999)

    parser.add_argument('--mode', type=str, default=None)
    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--retrain', action='store_true')

    parser.add_argument('--retrain-folder', type=str, default='retrain')
    parser.add_argument('--retrain-epochs', type=int, default=None)

    parser.add_argument('--dataset', type=str, default='voc')
    parser.add_argument('--cam-threshold', type=float, default=None)

    parser.add_argument('--ms-cam', action='store_true')
    parser.add_argument('--use-pl-seg', action='store_true')
    parser.add_argument('--use-attn-grad', action='store_true')
    parser.add_argument('--use-entropy', action='store_true')
    parser.add_argument('--l2norm', action='store_true')
    parser.add_argument('--skip-generation', action='store_true')
    parser.add_argument('--plain-training', action='store_true')
    parser.add_argument('--use-bal-weight', action='store_true')
    parser.add_argument('--no-crf', action='store_true')
    parser.add_argument('--no-merge', action='store_true')
    parser.add_argument('--use-mean-loss', action='store_true')
    parser.add_argument('--use-pix-mem', action='store_true')

    parser.add_argument('--test', action='store_true')
    parser.add_argument('--use-adam', action='store_true')
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--test-suffix', type=str, default=None)
    parser.add_argument('--retrain-suffix', type=str, default=None)
    parser.add_argument('--sal-weight', type=float, default=0.5)
    parser.add_argument('--custom-flag', type=int, default=None)
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--use-val', action='store_true')

    args = parser.parse_args()
    args = _set_default_roots(args)
    args.num_classes = {'voc': 21, 'coco': 81}[args.dataset]
    if args.dataset == 'coco':
        args.log_frequency = 82_081 // args.batch_size // 25
    args.train_size = {'voc':'321,321', 'coco': '417,417'}[args.dataset]
    args.cam_threshold = ({'voc': 0.1, 'coco': 0.3}[args.dataset]) if args.cam_threshold is None else args.cam_threshold
    args.seed_2stage_root = None


    #run(args, 'trainCAM')
    args.cam_threshold = 0.1
    args.seg_threshold = 0.1

    #run(args, 'genCAM')
    #run(args, 'trainSEG')
    #run(args, 'evalSEG')
    #run(args, 'trainE2E')
    #run(args, 'evalE2E')

    if args.plain_training:
        run(args, 'trainSEG')
        run(args, 'evalSEG')
    elif args.retrain:
        # Twostage Retrain
        set_num_epochs = args.num_epochs
        args.num_epochs = 30
        if not args.skip_generation:
            run(args, 'evalMCIS_MEM')
        args.num_epochs = set_num_epochs
        # run(args, 'trainMCIS_2STAGE')
        # run(args, 'evalMCIS_2STAGE')
        if args.use_val:
            subset = {'voc': 'train_aug_val', 'coco': 'train'}[args.dataset]
        else:
            subset = {'voc': 'train_aug', 'coco': 'train'}[args.dataset]
        args.seed_root = os.path.join(args.snapshot, get_snapshot_name(args), 'results', 'seedsMEMCRF', subset)
        run(args, 'trainSEG')
        run(args, 'evalSEG')
        run(args, 'testSEG')
    elif args.test:
        run(args, 'testMCIS')
    else:
        # Onestage Training
        run(args, 'trainMCIS')
        run(args, 'evalMCIS')
        run(args, 'testMCIS')


