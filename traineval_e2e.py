from misc import *
from core.model.vgg import *

def _compute_bal_weight(dataset, neg_ratio=4):
    if dataset == 'voc':
        filename = 'core/data/VOC/resources/train_aug.txt'
    elif dataset == 'coco':
        filename = '/home/junsong_fan/Data/Dataset/COCO/converted_labels/train2014.txt'
    else:
        raise RuntimeError(dataset)

    with open(filename) as f:
        data = list(f.readlines())
    if dataset == 'voc':
        ann_root = '/home/junsong_fan/Data/Dataset/VOC2012/Annotations'
        labels = [os.path.join(ann_root, name.strip()+'.xml') for name in data]
    elif dataset == 'coco':
        labels = [int(L) for x in data for L in x.strip().split(' ')[1:]]

    nImages = len(labels)
    counts = np.bincount(labels).astype(np.float64)
    pos = counts / nImages
    ratio = pos / (1 - pos) * neg_ratio
    out = torch.from_numpy(ratio.astype(np.float32))
    return out

def _criteria_balance_loss(logit, label, weight=None):
    #loss = - F.logsigmoid(logit) * label - F.logsigmoid(-logit) * (1 - label)
    #if weight is not None:
    #    loss = loss * label + loss * (1 - label) * weight.view(1, -1)

    #loss_red = (loss.sum(1) / (label.sum(1) + 1e-5)).mean()

    loss = F.multilabel_soft_margin_loss(logit, label, reduction='none')
    loss_red = (loss / (label.sum(1) + 1e-5)).mean()
    return loss_red

def _criteria_cross_entropy_2d(logit, label):
    assert label.dim() == 3, label.size()
    C = logit.size(1)
    label_oh = F.one_hot(label, 256)[..., :C].permute(0, 3, 1, 2).float()
    label_oh = (resize_maybe(label_oh, logit) > 0.5).float()

    non_ignore = label_oh.max(1, keepdim=True)[0]
    loss = - label_oh * F.log_softmax(logit, 1) * non_ignore
    loss_red = loss.sum() / torch.clamp_min(non_ignore.sum(), 1)
    return loss_red

@torch.no_grad()
def compute_seed_from_cam(logit, label_cls, saliency, threshold):
    logit = logit.detach()
    logit_max2d = logit.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
    cam = torch.clamp_min(logit, 0) / torch.clamp_min(logit_max2d, 1e-5)
    cam = cam * label_cls.unsqueeze(2).unsqueeze(3)

    saliency = resize_maybe(saliency.unsqueeze(1).float(), cam)
    bg = 1 - saliency / 255.

    cam_bg = torch.cat([bg, cam], 1)
    max_val, seed = cam_bg.max(1)
    seed[max_val < threshold] = 255
    return seed, cam_bg

@torch.no_grad()
def compute_seed_from_seg(logit, label_cls, saliency, threshold):
    logit = logit.detach()
    logit = torch.softmax(logit, 1)
    logit[:, 1:] = logit[:, 1:] * label_cls.unsqueeze(2).unsqueeze(3)
    if saliency is not None:
        saliency = resize_maybe(saliency.unsqueeze(1).float(), logit)
        bg = ((1 - saliency / 255.) + logit[:, 0:1]) / 2
        logit = torch.cat([bg, logit[:, 1:]], 1)

    max_val, seed = logit.max(1)
    seed[max_val < threshold] = 255
    return seed, logit

def run_trainE2E(gpu, args, port):
    is_distributed, world_size = config_env(gpu, args, port)
    is_log = gpu == 0
    args.snapshot = os.path.join(args.snapshot, get_snapshot_name(args))

    # Data
    subset = {'voc': 'train_aug', 'coco': 'train'}[args.dataset]
    train_loader, sampler = get_loader(args, subset, True, [args.seed_root, args.gt_root])

    # Model
    torch.manual_seed(args.seed)
    mod = VGG16_Sibling(
        args.num_classes,
        use_aspp=False,
        largefov_dilation=12
    ).to(gpu)
    mod.load_pretrained(args.pretrained, strict=not args.non_strict)

    lr_mult_params = mod.get_param_groups()
    optimizer = optim.SGD([{'params': p, 'lr': args.lr * lm} for lm, p in lr_mult_params.items()],
            momentum=args.sgd_mom, weight_decay=args.sgd_wd)

    if is_distributed:
        mod = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[gpu], find_unused_parameters=False)

    # Log
    if is_log:
        logger = getLogger(args.snapshot, args.model)
        summaryArgs(logger, args, 'green')
        meter = AvgMeter()
        saver = SaveParams(mod, args.snapshot, args.model)
        tb_writer = TensorBoardWriter(os.path.join(args.snapshot, 'runs'))
        meter.bind_tb_writer(tb_writer)
        DataPalette = eval(f'{args.dataset.upper()}.palette')
        get_segment_map = lambda x: DataPalette[x.ravel()].reshape(x.shape + (3,))
        timer = Timer()

    lr_scheduler = LrScheduler('poly', args.lr, {'power': 0.9, 'num_epochs': args.num_epochs})
    num_iter = len(train_loader)
    rampup_scheduler = RampUpScheduler(5 * num_iter, 2)

    # Train
    bal_weight = None
    for epoch in range(args.num_epochs):
        mod.train()
        freeze_bn(mod)
        if is_distributed:
            sampler.set_epoch(epoch)

        lr = lr_scheduler.get(max(epoch, 0))
        for lr_mult, param_group in zip(lr_mult_params.keys(), optimizer.param_groups):
            param_group['lr'] = lr * lr_mult
            if is_log:
                info(logger, f"Set lr={lr*lr_mult} for {len(param_group['params'])} params.", 'yellow')

        if is_log:
            meter.clean()
            examples = []
            vid = 0

        for batch, data in enumerate(train_loader, 1):
            optimizer.zero_grad()
            image, label_cls, label_sal, gt = data
            label_cls = label_cls[:, 1:].to(gpu)

            logit_cam, logit_seg = mod(image.to(gpu))
            loss_cam = _criteria_balance_loss(logit_cam.mean((2, 3)), label_cls, bal_weight)

            #assert args.cam_threshold == 0.1, args.cam_threshold
            seed, cam = compute_seed_from_cam(logit_cam, label_cls, label_sal.to(gpu), args.cam_threshold)
            loss_seg = _criteria_cross_entropy_2d(logit_seg, seed)

            rampup = rampup_scheduler.next()
            loss = loss_cam + rampup * loss_seg

            if args.use_pl_seg:
                seed_seg, seg = compute_seed_from_seg(logit_seg, label_cls, label_sal.to(gpu), 0.1)
                loss_seg2 = _criteria_cross_entropy_2d(logit_seg, seed_seg)
                loss = loss + rampup * loss_seg2

            loss.backward()
            optimizer.step()

            # Monitor
            if is_log and (batch % args.log_frequency == 0):
                meter.put('lr/rampup', rampup)
                meter.put('loss/cam', loss_cam.item())
                meter.put('loss/seg', loss_seg.item())

                vimg = add_image_mean(image[vid].data.cpu().numpy())
                vgt = get_segment_map(gt[vid].data.cpu().numpy())
                vseed = get_segment_map(seed[vid].data.cpu().numpy())
                vpred = get_segment_map(logit_seg[vid].argmax(0).data.cpu().numpy())

                vcam = get_score_map(cam[vid][1:].max(0)[0].data.cpu().numpy(), vimg)
                vccam = get_color_cam(vimg, cam[vid][1:], DataPalette)
                vseg = get_score_map(logit_seg[vid][1:].max(0)[0].data.cpu().numpy(), vimg)
                vcseg = get_color_cam(vimg, torch.softmax(logit_seg[vid], 0)[1:], DataPalette)

                examples_ = [vimg, vgt, vseed, vpred, vcam, vccam, vseg, vcseg]

                if args.use_pl_seg:
                    meter.put('loss/seg2', loss_seg2.item())
                    vseed2 = get_segment_map(seed_seg[vid].data.cpu().numpy())
                    examples_.insert(3, vseed2)

                examples.append(imhstack(examples_, height=240))
                imwrite(os.path.join(args.snapshot, 'train_demo_preview.jpg'), examples[-1])
                vid = (vid + 1) % (args.batch_size // world_size)

                timer.record()
                info(logger, f"Epoch={epoch}, Batch={batch}, {meter}, Speed={args.log_frequency*args.batch_size/timer.interval():.1f}")

        # Log & save
        if is_log:
            examples = imvstack(examples)
            imwrite(os.path.join(args.snapshot, 'train_demo', f'{args.model}-{epoch:04d}.jpg'), examples)
            tb_writer.add_image('image/demo', examples)
            saver(epoch)
            info(logger, f'Saved params to: {saver.filename}', 'yellow')
            tb_writer.flush()

    if is_log:
        tb_writer.close()
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def run_evalE2E(gpu, args, port):
    is_distributed, world_size = config_env(gpu, args, port)
    is_log = gpu == 0
    args.snapshot = args.force_snapshot if args.force_snapshot else os.path.join(args.snapshot, get_snapshot_name(args))

    # Data
    subset = 'val'
    loader, _ = get_loader(args, subset, False, return_src=True)

    # Model
    torch.manual_seed(args.seed)
    mod = VGG16_Sibling(
        args.num_classes,
        use_aspp=False,
        largefov_dilation=12,
        mode='eval'
    ).to(gpu)
    if is_distributed:
        mod = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[gpu], find_unused_parameters=False)

    pretrained = os.path.join(args.snapshot, f'checkpoint/{args.model}-{args.num_epochs-1:04d}.pth')
    assert os.path.exists(pretrained), pretrained
    info(None, f'Using pretrained params: {pretrained}', 'red')
    pretrained = torch.load(pretrained)
    mod.load_state_dict(pretrained, strict=True)

    save_root = os.path.join(args.snapshot, 'results', 'seeds', subset)
    os.makedirs(save_root, exist_ok=True)
    save_crf_root = os.path.join(args.snapshot, 'results', 'seedsCRF', subset)
    os.makedirs(save_crf_root, exist_ok=True)

    # Inter
    mod.eval()
    pad_size = {'voc': 513, 'coco': 641}[args.dataset]
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

    compute_some_iou = eval(f'compute_{args.dataset.lower()}_iou')
    gt_root = args.gt_root if 'train' in subset else args.gt_val_root
    if is_log and (subset != 'test'):
        logger = getLogger(args.snapshot, args.model)
        tb_writer = TensorBoardWriter(os.path.join(args.snapshot, 'runs'))
        iou = compute_some_iou(f'{save_root}', gt_root, subset, logger=logger)
        tb_writer.add_scalar(f'mIOU/{subset}/noCRF', iou.mean())

        iou = compute_some_iou(f'{save_crf_root}', gt_root, subset, logger=logger)
        tb_writer.add_scalar(f'mIOU/{subset}/CRF', iou.mean())

        tb_writer.flush()
        tb_writer.close()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

