from misc import *
from core.model.mcis_model import * 

import argparse
import copy

def run_evalTest(gpu, args, port):
    is_distributed, world_size = config_evn(gpu, args, port)
    is_log = gpu == 0
    assert os.path.exists(args.checkpoint_file), args.checkpoint_file

    # Data
    subset = 'test'
    loader, _ = get_loader(args, subset, False, return_src=True)

    # Model
    torch.manual_seed(args.seed)
    backboner = get_backbone(args.model, args.num_classes)
    if args.use_mcis:
        mod = MCIS(
            backboner,
            args.threshold,
            args.capacity,
            stop_attn_grad=not args.use_attn_grad,
            use_ema=args.use_ema,
            ema_momentum=args.ema_momentum,
            mode='eval'
        ).to(gpu)
    else:
        mod = backboner().to(gpu)
    if is_distributed:
        mod = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[gpu], find_unused_parameters=False)

    info(None, f'Using pretrained params: {args.checkpoint_file}', 'red')
    pretrained = torch.load(args.checkpoint_file)
    mod.load_state_dict(pretrained, strict=True)

    save_root = os.path.join(args.target_folder, 'noCRF', f'results/VOC2012/Segmentation/comp5_{subset}_cls')
    save_crf_root = os.path.join(args.target_folder, 'CRF', f'results/VOC2012/Segmentation/comp5_{subset}_cls')
    if is_log:
        os.makedirs(save_root, exist_ok=True)

    # Infer
    mod.eval()
    pad_size = 513
    test_scales = [float(x) for x in args.test_scales.split(',')]
    pool = mp.Pool(32)
    crf_jobs = []
    with torch.no_grad():
        for image, label_cls, src in loader:
            assert image.shape[0] == 1, image.shape
            image = image.to(gpu)

            h, w = image.shape[2:]
            ph, pw = pad_size - h, pad_size - w
            if ph > 0 or pw > 0:
                image = F.pad(image, (0, pw, 0, ph), 'constant', 0)

            pred_max = None
            pred_ms = []
            for s in test_scales:
                if s == 1:
                    img_ = image
                else:
                    size = int((pad_size - 1) * s) + 1
                    img_ = F.interpolate(image, (size, size), mode='bilinear', align_corners=True)
                img_flip = torch.flip(img_, (3,))
                img_in = torch.cat([img_, img_flip], 0)
                pred = mod(img_in)
                pred = resize_maybe(pred, (pad_size, pad_size))

                pred_max = pred if pred_max is None else torch.maximum(pred, pred_max)
                pred_ms.append(pred.data.cpu().numpy())
            pred_ms.append(pred_max.data.cpu().numpy())

            pred_ms = np.array(pred_ms).mean(0)
            pred_ms = (pred_ms[0] + pred_ms[1, :, :, ::-1]) / 2
            pred_ms = pred_ms[:, :h, :w]
    
            seed = pred_ms.argmax(0).astype(np.uint8)
            name = os.path.basename(src[0]).rsplit('.', 1)[0]
            cv2.imwrite(os.path.join(save_root, name+'.png'), seed)

            if not args.no_crf:
                crf_jobs.append(pool.apply_async(crf_postprocessing, (
                        src[0],
                        pred_ms.astype(np.float16),
                        os.path.join(save_crf_root, name+'.png')
                    )
                ))
            if len(crf_jobs) == 1:
                crf_jobs[0].get()

    [job.get() for job in crf_jobs[1:]]

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


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
    img = '/home/junsong_fan/diskf/data/VOC2012/JPEGImages'
    gt = '/home/junsong_fan/diskf/data/VOC2012/extra/SegmentationClassAug'
    sal='./seeds/saliency_aug'
    ann='/home/junsong_fan/diskf/data/VOC2012/Annotations'
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

    parser.add_argument('--use-mcis', action='store_true')
    parser.add_argument('--checkpoint-file', type=str, required=True)
    parser.add_argument('--target-folder', type=str, required=True)

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

    run(args, 'evalTest')