from misc import *
from core.model.vgg import *


def _compute_bal_weight(dataset, neg_ratio=2):
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
    loss = - F.logsigmoid(logit) * label - F.logsigmoid(-logit) * (1 - label)
    if weight is not None:
        loss = loss * label + loss * (1 - label) * weight.view(1, -1)

    loss_red = (loss.sum(1) / torch.clamp_min(label.sum(1), 1)).mean()
    return loss_red

def run_trainCAM(gpu, args, port):
    is_distributed, world_size = config_env(gpu, args, port)
    is_log = gpu == 0
    args.snapshot = os.path.join(args.snapshot, get_snapshot_name(args))

	# Data
    subset = {'voc': 'train_aug', 'coco': 'train'}[args.dataset]
    train_loader, sampler = get_loader(args, subset, True, args.gt_root)

	# Model
    torch.manual_seed(args.seed)
    mod = VGG16(args.num_classes - 1,
        use_aspp=False,
        largefov_dilation=1
    ).to(gpu)
    mod.load_pretrained(args.pretrained, strict=not args.non_strict)

    lr_mult_params = mod.get_param_groups()
    optimizer = optim.SGD([{'params': p, 'lr': args.lr * lm} for lm, p in lr_mult_params.items()],
            momentum=args.sgd_mom, weight_decay=args.sgd_wd)

    if is_distributed:
        mod = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[gpu], find_unused_parameters=True)

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
            image, label_cls, gt = data
            label_cls = label_cls[:, 1:]
            assert label_cls.shape[1] == args.num_classes - 1

            logit = mod(image.to(gpu))
            logit_gap = logit.mean((2, 3))
            loss = _criteria_balance_loss(logit_gap, label_cls.to(gpu), bal_weight)

            loss.backward()
            optimizer.step()

            # Monitor
            if is_log and (batch % args.log_frequency == 0):
                meter.put('loss/cam', loss.item())

                vimg = add_image_mean(image[vid].data.cpu().numpy())
                vgt = get_segment_map(gt[vid].data.cpu().numpy())

                cam = torch.clamp_min(logit, 0) * label_cls.unsqueeze(2).unsqueeze(3).to(gpu)
                vcam = get_score_map(cam[vid].max(0)[0].data.cpu().numpy(), vimg)
                vccam = get_color_cam(vimg, cam[vid], DataPalette)
                examples_ = [vimg, vgt, vcam, vccam]

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


def run_genCAM(gpu, args, port):
    is_distributed, world_size = config_env(gpu, args, port)
    is_log = gpu == 0
    args.snapshot = os.path.join(args.snapshot, get_snapshot_name(args))

	# Data
    subset = {'voc': 'train_aug', 'coco': 'train'}[args.dataset]
    loader, _ = get_loader(args, subset, False, args.seed_root, return_src=True)

    # Model
    torch.manual_seed(args.seed)
    mod = VGG16(args.num_classes - 1,
        use_aspp=False,
        largefov_dilation=1
    ).to(gpu)
    if is_distributed:
        mod = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[gpu], find_unused_parameters=True)

    pretrained = os.path.join(args.snapshot, f'checkpoint/{args.model}-{args.num_epochs-1:04d}.pth')
    assert os.path.exists(pretrained), pretrained
    info(None, f'Using pretrained params: {pretrained}', 'red')
    pretrained = torch.load(pretrained)
    mod.load_state_dict(pretrained, strict=True)

    save_root = os.path.join(args.snapshot, 'results', 'seeds', subset)
    os.makedirs(save_root, exist_ok=True)

    # Infer
    mod.eval()
    with torch.no_grad():
        for image, label_cls, label_sal, src in loader:
            assert image.shape[0] == 1, image.shape

            label_cls = label_cls[:, 1:]
            logit = mod(image.to(gpu))
            logit_max2d = logit.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
            cam = torch.clamp_min(logit, 0) / torch.clamp_min(logit_max2d, 1e-5)
            cam = cam * label_cls.unsqueeze(2).unsqueeze(3).to(gpu)

            h, w = image.shape[2:]
            cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True)

            bg = label_sal.to(gpu).unsqueeze(1)
            #bg = (1 - bg / 255.) * 0.3
            bg = 1 - bg / 255.

            cam_bg = torch.cat([bg, cam], 1)[0]

            max_val, seed = cam_bg.max(0)
            seed[max_val < args.cam_threshold] = 255

            seed = seed.data.cpu().numpy().astype(np.uint8)
            name = os.path.basename(src[0]).rsplit('.', 1)[0]
            cv2.imwrite(os.path.join(save_root, name + '.png'), seed)

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()
