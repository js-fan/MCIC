from misc import *
#from core.model.vgg import *
from core.model.mcis_model import *

def _criteria_cross_entropy_2d(logit, label):
    assert label.dim() == 3, label.size()
    C = logit.size(1)
    label_oh = F.one_hot(label, 256)[..., :C].permute(0, 3, 1, 2).float()
    label_oh = (resize_maybe(label_oh, logit) > 0.5).float()

    non_ignore = label_oh.max(1, keepdim=True)[0]
    loss = - label_oh * F.log_softmax(logit, 1) * non_ignore
    loss_red = loss.sum() / torch.clamp_min(non_ignore.sum(), 1)
    return loss_red

def run_trainSEG(gpu, args, port):
    is_distributed, world_size = config_env(gpu, args, port)
    is_log = gpu == 0
    args.snapshot = os.path.join(args.snapshot, get_snapshot_name(args))
    if args.retrain:
        retrain_suffix = f'_{args.retrain_suffix}' if args.retrain_suffix is not None else ''
        args.snapshot = os.path.join(args.snapshot, f'twoStagePlain{args.num_epochs}EP{retrain_suffix}')

	# Data
    if args.use_val:
        subset = {'voc': 'train_aug_val', 'coco': 'train'}[args.dataset]
    else:
        subset = {'voc': 'train_aug', 'coco': 'train'}[args.dataset]
    train_loader, sampler = get_loader(args, subset, True, [args.seed_root, args.gt_root])

    # Model
    torch.manual_seed(args.seed)
    backboner = get_backbone('plain_' + args.model, args.num_classes, args.dropout)
    mod = backboner().to(gpu)
    # mod = VGG16(
    #     args.num_classes,
    #     use_aspp=False,
    #     largefov_dilation=12
    # ).to(gpu)
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

    # Train
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
            image, label_cls, seed, gt = data

            logit = mod(image.to(gpu))
            loss = _criteria_cross_entropy_2d(logit, seed.to(gpu))

            loss.backward()
            optimizer.step()

            # Monitor
            if is_log and (batch % args.batch_size == 0):
                meter.put('loss/seg', loss.item())

                vimg = add_image_mean(image[vid].data.cpu().numpy())
                vgt = get_segment_map(gt[vid].data.cpu().numpy())
                vseed = get_segment_map(seed[vid].data.cpu().numpy())
                vpred = get_segment_map(logit[vid].argmax(0).data.cpu().numpy())

                #cam = torch.clamp_min(logit[vid][1:], 0)
                cam = torch.softmax(logit[vid], 0)[1:]
                cam_filter = cam * label_cls[vid][1:].to(gpu).view(-1, 1, 1)

                vcam = get_score_map(cam.max(0)[0].data.cpu().numpy(), vimg)
                vccam = get_color_cam(vimg, cam, DataPalette)
                vcam_filter = get_score_map(cam_filter.max(0)[0].data.cpu().numpy(), vimg)
                vccam_filter = get_color_cam(vimg, cam_filter, DataPalette)
                examples_ = [vimg, vgt, vseed, vpred, vcam, vccam, vcam_filter, vccam_filter]

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


def run_testSEG(gpu, args, port):
    return run_evalSEG(gpu, args, port, 'test')

def run_evalSEG(gpu, args, port, subset='val'):
    is_distributed, world_size = config_env(gpu, args, port)
    is_log = gpu == 0
    args.snapshot = os.path.join(args.snapshot, get_snapshot_name(args))
    if args.retrain:
        retrain_suffix = f'_{args.retrain_suffix}' if args.retrain_suffix is not None else ''
        args.snapshot = os.path.join(args.snapshot, f'twoStagePlain{args.num_epochs}EP{retrain_suffix}')

    # Data
    loader, _ = get_loader(args, subset, False, return_src=True)

    # Model
    torch.manual_seed(args.seed)
    backboner = get_backbone('plain_' + args.model, args.num_classes, args.dropout)
    mod = backboner().to(gpu)
    # mod = VGG16(
    #     args.num_classes, 
    #     use_aspp=False,
    #     largefov_dilation=12
    # ).to(gpu)
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

