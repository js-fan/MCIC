from misc import *
from core.model.vgg import *
from core.model.mcis_model import * 

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
    ratio = pos / (1 - pos) / neg_ratio
    out = torch.from_numpy(ratio.astype(np.float32))
    return out

def _criteria_balance_loss(logit, label, weight=None):
    #loss = - F.logsigmoid(logit) * label - F.logsigmoid(-logit) * (1 - label)
    #if weight is not None:
    #    loss = loss * label + loss * (1 - label) * weight.view(1, -1)

    #loss_red = (loss.sum(1) / (label.sum(1) + 1e-5)).mean()
    if weight is not None:
        loss = - F.logsigmoid(logit) * label - F.logsigmoid(-logit) * (1 - label) * weight.view(1, -1)
        loss = loss.sum(1)
    else:
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

def _criteria_entroy(logit, tau=1):
    logit = logit / tau
    softmax = torch.softmax(logit, -1)
    log_softmax = torch.log(softmax + 1e-5)
    loss = - (softmax * log_softmax)
    loss_red = loss.sum(-1).mean()
    return loss_red

@torch.no_grad()
def compute_seed_from_cam(logit, label_cls, saliency, threshold):
    logit = logit.detach()
    logit_max2d = logit.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
    cam = torch.clamp_min(logit, 0) / torch.clamp_min(logit_max2d, 1e-5)
    cam = cam * label_cls.unsqueeze(2).unsqueeze(3)

    saliency = resize_maybe(saliency.unsqueeze(1).float(), cam)
    #bg = 1 - saliency / 255.
    bg = (saliency / 255 < threshold).float()

    cam_bg = torch.cat([bg, cam], 1)
    max_val, seed = cam_bg.max(1)
    seed[max_val < threshold] = 255
    return seed, cam_bg

@torch.no_grad()
def compute_seed_from_seg(logit, label_cls, saliency, threshold, sal_weight=None):
    logit = logit.detach()
    logit = torch.softmax(logit, 1)
    logit[:, 1:] = logit[:, 1:] * label_cls.unsqueeze(2).unsqueeze(3)
    if saliency is not None:
        saliency = resize_maybe(saliency.unsqueeze(1).float(), logit)
        # bg = ((1 - saliency / 255.) + logit[:, 0:1]) / 2
        bg = (1 - saliency / 255.) * sal_weight + logit[:, 0:1] * (1 - sal_weight)
        logit = torch.cat([bg, logit[:, 1:]], 1)

    max_val, seed = logit.max(1)
    seed[max_val < threshold] = 255
    return seed, logit

@torch.no_grad()
def compute_merge_seed(seeda, seedb, num_classes):
    seeda_oh = F.one_hot(seeda, 256)[..., :num_classes]
    seedb_oh = F.one_hot(seedb, 256)[..., :num_classes]

    fg_union = torch.maximum(seeda_oh[..., 1:], seedb_oh[..., 1:])
    bg_inter = torch.minimum(seeda_oh[..., 0:1], seedb_oh[..., 0:1])
    seedm_oh = torch.cat([bg_inter, fg_union], -1)
    seedm = seedm_oh.argmax(-1)
    seedm[seedm_oh.sum(-1) != 1] = 255
    return seedm

def run_trainMCIS(gpu, args, port):
    is_distributed, world_size = config_env(gpu, args, port)
    is_log = gpu == 0
    args.snapshot = os.path.join(args.snapshot, get_snapshot_name(args))

    # Data
    if args.use_val:
        subset = {'voc': 'train_aug_val', 'coco': 'train'}[args.dataset]
    else:
        subset = {'voc': 'train_aug', 'coco': 'train'}[args.dataset]
    train_loader, sampler = get_loader(args, subset, True, [args.seed_root, args.gt_root])

    # Model
    torch.manual_seed(args.seed)
    backboner = get_backbone(args.model, args.num_classes, args.dropout)
    mod = MCIS(
        backboner,
        args.threshold,
        args.capacity,
        stop_attn_grad=not args.use_attn_grad,
        use_ema=args.use_ema,
        ema_momentum=args.ema_momentum,
        mode='mcis_pix' if args.use_pix_mem else 'mcis',
        l2norm=args.l2norm
    ).to(gpu)
    mod.load_pretrained(args.pretrained, strict=not args.non_strict)

    lr_mult_params = mod.get_param_groups()

    if args.use_adam:
        optimizer = optim.AdamW([{'params': p, 'lr': args.lr * lm} for lm, p in lr_mult_params.items()],
                weight_decay=args.sgd_wd)
    else:
        optimizer = optim.SGD([{'params': p, 'lr': args.lr * lm} for lm, p in lr_mult_params.items()],
                momentum=args.sgd_mom, weight_decay=args.sgd_wd)

    if is_distributed:
        mod = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[gpu], find_unused_parameters=args.use_ema)

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
    bal_weight = _compute_bal_weight(args.dataset).to(gpu) if args.use_bal_weight else None
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
            label_cls[:, 0] = 1
            label_cls = label_cls.to(gpu)
            label_sal = label_sal.to(gpu)

            logit_cam, logit_seg, logit_aseg, attn_w, *others = mod(image.to(gpu), label_cls)

            # pseudo labels
            label_cls_ = label_cls[:, 1:]
            seed, cam = compute_seed_from_cam(logit_cam, label_cls_, label_sal, args.cam_threshold)
            #seed_seg, seg = compute_seed_from_seg(logit_seg, label_cls, label_sal, args.seg_threshold)
            seed_aseg, aseg = compute_seed_from_seg(logit_aseg, label_cls_, label_sal, args.seg_threshold, args.sal_weight)
            seed_merge = compute_merge_seed(seed, seed_aseg, args.num_classes)

            # loss-cam
            loss_cam = _criteria_balance_loss(logit_cam.mean((2, 3)), label_cls[:, 1:], bal_weight)

            # loss-seg
            if args.no_merge:
                seed_list = [seed, seed_aseg]
            else:
                seed_list = [seed, seed_aseg, seed_merge]
            #seed_list = [seed_merge]

            rampup = rampup_scheduler.next()

            loss_seg_list, loss_aseg_list = [], []
            for ithseed, _seed in enumerate(seed_list):
                scale = (1 - rampup) if ithseed == 0 else 1
                scale = max(scale, 0.6)
                #if args.custom_flag == 0:
                #    scale = (1 - rampup) if ithseed == 0 else 1
                #    scale = max(scale, 0.5)
                #elif args.custom_flag == 1:
                #    scale = (1 - rampup) if ithseed == 0 else 1
                #    scale = max(scale, 0.7)
                #else:
                #    raise RuntimeError
                loss_seg_list.append(_criteria_cross_entropy_2d(logit_seg, _seed) * scale)
                loss_aseg_list.append(_criteria_cross_entropy_2d(logit_aseg, _seed) * scale)
            loss_seg = sum(loss_seg_list + loss_aseg_list) 
            if args.use_mean_loss:
                loss_seg = loss_seg / len(seed_list)

            loss = loss_cam + rampup * loss_seg

            # loss-entropy
            if args.use_entropy:
                attn_logit, mem_lbl = others
                loss_ent = _criteria_entroy(attn_logit)
                loss = loss + (rampup * 1) * loss_ent

            loss.backward()
            optimizer.step()

            # Monitor
            if is_log and (batch % args.log_frequency == 0):
                meter.put('lr/rampup', rampup)
                meter.put('loss/cam', loss_cam.item())
                for i, _loss_seg in enumerate(loss_seg_list):
                    meter.put(f'loss/seg{i}', _loss_seg.item())
                for i, _loss_aseg in enumerate(loss_aseg_list):
                    meter.put(f'loss/aseg{i}', _loss_aseg.item())
                if args.use_entropy:
                    meter.put(f'loss/ent', loss_ent.item())

                vimg = add_image_mean(image[vid].data.cpu().numpy())
                vgt = get_segment_map(gt[vid].data.cpu().numpy())
                examples_ = [vimg, vgt]

                for _seed in seed_list:
                    vseed = get_segment_map(_seed[vid].data.cpu().numpy())
                    examples_.append(vseed)

                seed_prob_list = [cam, aseg]
                for _prob in seed_prob_list:
                    vprob = get_score_map(_prob[vid][1:].max(0)[0].data.cpu().numpy(), vimg)
                    vcprob = get_color_cam(vimg, _prob[vid][1:], DataPalette)
                    examples_ += [vprob, vcprob]

                vattn_w = get_score_map(attn_w[vid].sum(0).view(*logit_aseg.shape[2:]).data.cpu().numpy(), vimg)
                examples_.append(vattn_w)

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

def run_testMCIS(gpu, args, port):
    return run_evalMCIS(gpu, args, port, 'test')

def run_evalMCIS(gpu, args, port, subset='val'):
    is_distributed, world_size = config_env(gpu, args, port)
    is_log = gpu == 0
    args.snapshot = args.force_snapshot if args.force_snapshot else os.path.join(args.snapshot, get_snapshot_name(args))

    # Data
    #subset = 'val'
    loader, _ = get_loader(args, subset, False, return_src=True)

    # Model
    torch.manual_seed(args.seed)
    backboner = get_backbone(args.model, args.num_classes, args.dropout)
    mod = MCIS(
        backboner,
        args.threshold,
        args.capacity,
        stop_attn_grad=not args.use_attn_grad,
        use_ema=args.use_ema,
        ema_momentum=args.ema_momentum,
        mode='eval',
    ).to(gpu)
    if is_distributed:
        mod = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[gpu], find_unused_parameters=False)

    pretrained = os.path.join(args.snapshot, f'checkpoint/{args.model}-{args.num_epochs-1:04d}.pth')
    assert os.path.exists(pretrained), pretrained
    info(None, f'Using pretrained params: {pretrained}', 'red')
    pretrained = torch.load(pretrained)
    mod.load_state_dict(pretrained, strict=True)

    save_root = os.path.join(args.snapshot, 'results', 'seeds', 
            subset if args.test_suffix is None else f'{subset}_{args.test_suffix}')

    os.makedirs(save_root, exist_ok=True)
    save_crf_root = os.path.join(args.snapshot, 'results', 'seedsCRF',
            subset if args.test_suffix is None else f'{subset}_{args.test_suffix}')
    os.makedirs(save_crf_root, exist_ok=True)

    # Inter
    mod.eval()
    pad_size = {'voc': 513, 'coco': 641}[args.dataset]
    test_scales = [float(x) for x in args.test_scales.split(',')]
    pool = mp.Pool(32)
    crf_jobs = []
    with torch.no_grad():
        #for image, label_cls, src in loader:
        for load_data in loader:
            if len(load_data) == 3:
                image, label_cls, src = load_data
            elif len(load_data) == 2:
                image, src = load_data
            else:
                raise RuntimeError(len(load_data))
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

    compute_some_iou = eval(f'compute_{args.dataset.lower()}_iou')
    gt_root = args.gt_root if 'train' in subset else args.gt_val_root
    if is_log and (subset != 'test'):
        logger = getLogger(args.snapshot, args.model)
        tb_writer = TensorBoardWriter(os.path.join(args.snapshot, 'runs'))
        iou = compute_some_iou(f'{save_root}', gt_root, subset, logger=logger)
        tb_writer.add_scalar(f'mIOU/{subset}/noCRF', iou.mean())

        if not args.no_crf:
            iou = compute_some_iou(f'{save_crf_root}', gt_root, subset, logger=logger)
            tb_writer.add_scalar(f'mIOU/{subset}/CRF', iou.mean())

        tb_writer.flush()
        tb_writer.close()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def run_evalMCIS_MEM(gpu, args, port):
    is_distributed, world_size = config_env(gpu, args, port)
    is_log = gpu == 0
    args.snapshot = args.force_snapshot if args.force_snapshot else os.path.join(args.snapshot, get_snapshot_name(args))
    
    # Data
    if args.use_val:
        subset = {'voc': 'train_aug_val', 'coco': 'train'}[args.dataset]
    else:
        subset = {'voc': 'train_aug', 'coco': 'train'}[args.dataset]
    loader, _ = get_loader(args, subset, False, return_src=True)

    # Model
    torch.manual_seed(args.seed)
    backboner = get_backbone(args.model, args.num_classes, args.dropout)
    mod = MCIS(
        backboner,
        args.threshold,
        args.capacity,
        stop_attn_grad=not args.use_attn_grad,
        use_ema=args.use_ema,
        ema_momentum=args.ema_momentum,
        mode='mcis_eval',
    ).to(gpu)
    if is_distributed:
        mod = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[gpu], find_unused_parameters=False)

    pretrained = os.path.join(args.snapshot, f'checkpoint/{args.model}-{args.num_epochs-1:04d}.pth')
    assert os.path.exists(pretrained), pretrained
    info(None, f'Using pretrained params: {pretrained}', 'red')
    pretrained = torch.load(pretrained)
    mod.load_state_dict(pretrained, strict=True)

    save_roots = [
        os.path.join(args.snapshot, 'results', 'seeds', subset),
        os.path.join(args.snapshot, 'results', 'seedsCRF', subset),
        os.path.join(args.snapshot, 'results', 'seedsMEM', subset),
        os.path.join(args.snapshot, 'results', 'seedsMEMCRF', subset)
    ]
    for save_root in save_roots:
        os.makedirs(save_root, exist_ok=True)

    # Infer
    mod.eval()
    pad_size = {'voc': 513, 'coco': 641}[args.dataset]
    test_scales = [float(x) for x in args.test_scales.split(',')]
    pool = mp.Pool(32)
    crf_jobs = []
    nPreds = len(save_roots) // 2
    with torch.no_grad():
        for image, label_cls, src in loader:
            assert image.shape[0] == 1, image.shape
            image = image.to(gpu)
            label_cls[:, 0] = 1
            label_cls = label_cls.to(gpu).repeat(2, 1)

            h, w = image.shape[2:]
            ph, pw = pad_size - h, pad_size - w
            if ph > 0 or pw > 0 :
                image = F.pad(image, (0, pw, 0, ph), 'constant', 0)

            preds_max = [None for _ in range(nPreds)]
            preds_ms = [[] for _ in range(nPreds)]
            for s in test_scales:
                if s == 1:
                    img_ = image
                else:
                    size = int((pad_size - 1) * s) + 1
                    img_ = F.interpolate(image, (size, size), mode='bilinear', align_corners=True)
                img_flip = torch.flip(img_, (3,))
                img_in = torch.cat([img_, img_flip], 0)
                preds = mod(img_in, label_cls)

                for i, pred in enumerate(preds):
                    pred = resize_maybe(pred, (pad_size, pad_size))
                    preds_max[i] = pred if preds_max[i] is None else torch.maximum(preds_max[i], pred)
                    preds_ms[i].append(pred.data.cpu().numpy())

            for i in range(nPreds):
                preds_ms[i].append(preds_max[i].data.cpu().numpy())

            preds_ms = np.array(preds_ms).mean(1)
            assert preds_ms.ndim == 5, preds_ms.shape
            preds_ms = (preds_ms[:, 0] + preds_ms[:, 1, :, :, ::-1]) / 2
            preds_ms = preds_ms[:, :, :h, :w]

            name = os.path.basename(src[0]).rsplit('.', 1)[0]
            for i, pred_ms in enumerate(preds_ms):
                seed = pred_ms.argmax(0).astype(np.uint8)
                cv2.imwrite(os.path.join(save_roots[i*2], name+'.png'), seed)

                crf_jobs.append(pool.apply_async(crf_postprocessing, (
                    src[0],
                    pred_ms.astype(np.float16),
                    os.path.join(save_roots[i*2 + 1], name+'.png')
                )))
                if len(crf_jobs) == 1:
                    crf_jobs[0].get()

    [job.get() for job in crf_jobs]
    if is_distributed:
        dist.barrier()

    compute_some_iou = eval(f'compute_{args.dataset.lower()}_iou')
    gt_root = args.gt_root if 'train' in subset else args.gt_val_root
    if is_log and (subset != 'test'):
        logger = getLogger(args.snapshot, args.model)
        tb_writer = TensorBoardWriter(os.path.join(args.snapshot, 'runs'))
        for save_root in save_roots:
            iou = compute_some_iou(f'{save_root}', gt_root, subset, logger=logger)
            tag = os.path.basename(os.path.dirname(save_root))
            tb_writer.add_scalar(f'mIOU/{tag}/{subset}', iou.mean())

        tb_writer.flush()
        tb_writer.close()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


