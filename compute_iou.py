import multiprocessing as mp
from core.utils import *
_curr_path = os.path.dirname(os.path.abspath(__file__))
_voc_data_list_root = os.path.join(_curr_path, 'core', 'data', 'VOC', 'resources')
_coco_data_list_root = os.path.join(_curr_path, 'core', 'data', 'COCO', 'resources')

__all__ = ['compute_voc_iou', 'compute_coco_iou', 'compute_ade_iou', 'compute_cs_iou']

def compute_voc_iou(pred_root, gt_root, data_list=None, logger=None):
    gt_dict = {x.rsplit('.', 1)[0]: os.path.join(gt_root, x) for x in os.listdir(gt_root) if x.endswith('.png')}
    pred_dict = {x.rsplit('.', 1)[0]: os.path.join(pred_root, x) for x in os.listdir(pred_root) if x.endswith('.png')}

    if data_list is not None:
        if data_list in ['train_aug', 'val', 'test']:
            data_list = os.path.join(_voc_data_list_root, data_list + '.txt')
        assert os.path.exists(data_list), data_list
        with open(data_list) as f:
            names = [x.strip().split(' ')[0] for x in f.readlines()]
    else:
        names = list(gt_dict.keys())

    for name in names:
        assert name in gt_dict, 'Cannot find gt: {}'.format(name)
        assert name in pred_dict, 'Cannot find pred: {}'.format(name)
        assert os.path.exists(gt_dict[name]), 'Cannot find gt: {}'.format(gt_dict[name])
        assert os.path.exists(pred_dict[name]), 'Cannot find pred: {}'.format(pred_dict[name])

    pairs = [[gt_dict[name], pred_dict[name]] for name in names]
    info(logger, '{}\nIn total {} samples'.format(pred_root, len(pairs)))

    # iou
    iou = compute_iou(pairs, 21)
    info(logger, 'VOC2012 mIoU: {}\n{}'.format(iou.mean(), iou))
    return iou


def compute_ade_iou(pred_root, gt_root, data_list=None, logger=None):
    gt_dict = {x.rsplit('.', 1)[0]: os.path.join(gt_root, x) for x in os.listdir(gt_root) if x.endswith('.png')}
    pred_dict = {x.rsplit('.', 1)[0]: os.path.join(pred_root, x) for x in os.listdir(pred_root) if x.endswith('.png')}

    if data_list is not None:
        with open(data_list) as f:
            names = [x.strip().split(' ')[0] for x in f.readlines()]
    else:
        names = list(gt_dict.keys())

    for name in names:
        assert name in gt_dict, 'Cannot find gt: {}'.format(name)
        assert name in pred_dict, 'Cannot find pred: {}'.format(name)
        assert os.path.exists(gt_dict[name]), 'Cannot find gt: {}'.format(gt_dict[name])
        assert os.path.exists(pred_dict[name]), 'Cannot find pred: {}'.format(pred_dict[name])

    pairs = [[gt_dict[name], pred_dict[name]] for name in names]
    info(logger, 'In total {} samples'.format(len(pairs)))

    # iou
    iou = compute_iou(pairs, 150, num_threads=16)
    info(logger, 'ADE20k mIou: {}\n{}'.format(iou.mean(), iou))
    return iou

def compute_cs_iou(pred_root, gt_root, data_list=None, logger=None):
    citys = os.listdir(gt_root)
    gt_dict = {}
    for city in citys:
        srcs = [x for x in os.listdir(os.path.join(gt_root, city)) if x.endswith('_labelIds.png')]
        gt_dict.update({'_'.join(src.split('_')[:3]) : os.path.join(gt_root, city, src) for src in srcs})

    pred_dict = {}
    citys = os.listdir(pred_root)
    for city in citys:
        if os.path.isdir(os.path.join(pred_root, city)):
            srcs = [x for x in os.listdir(os.path.join(pred_root, city))]
            pred_dict.update({'_'.join(src.split('_')[:3]) : os.path.join(pred_root, city, src) for src in srcs \
                    if src.endswith('.png')})
        else:
            src = city
            if src.rsplit('.', 1)[-1].lower() in ['png', 'jpg', 'jpeg']:
                pred_dict['_'.join(src.split('.')[0].split('_')[:3])] = os.path.join(pred_root, city)
    
    if data_list is not None:
        with open(data_list) as f:
            names = [x.strip().split(' ')[0] for x in f.readlines()]
    else:
        names = list(gt_dict.keys())

    for name in names:
        assert name in gt_dict, 'Cannot find gt: {}'.format(name)
        assert name in pred_dict, 'Cannot find pred: {}'.format(name)
        assert os.path.exists(gt_dict[name]), 'Cannot find gt: {}'.format(gt_dict[name])
        assert os.path.exists(pred_dict[name]), 'Cannot find pred: {}'.format(pred_dict[name])

    pairs = [[gt_dict[name], pred_dict[name]] for name in names]
    info(logger, 'In total {} samples'.format(len(pairs)))

    # class
    iou_cls = compute_iou(pairs, 19, map_func=CS.id2trainId, num_threads=16)
    info(logger, 'Class mIoU: {}\n{}'.format(iou_cls.mean(), iou_cls))

    # category
    iou_cat = compute_iou(pairs, 7, map_func=CS.id2catId, num_threads=16)
    info(logger, 'Category mIoU: {}\n{}'.format(iou_cat.mean(), iou_cat))
    return iou_cls, iou_cat

def compute_coco_iou(pred_root, gt_root, data_list=None, logger=None):
    gt_dict = {x.rsplit('.', 1)[0]: os.path.join(gt_root, x) for x in os.listdir(gt_root) if x.endswith('.png')}
    pred_dict = {x.rsplit('.', 1)[0]: os.path.join(pred_root, x) for x in os.listdir(pred_root) if x.endswith('.png')}

    if data_list is not None:
        data_list = os.path.join(_coco_data_list_root, data_list + '.txt')
        assert os.path.exists(data_list), data_list
        with open(data_list) as f:
            names = [x.strip().split(' ')[0].rsplit('.', 1)[0] for x in f.readlines()]
    else:
        names = list(gt_dict.keys())

    for name in names:
        assert name in gt_dict, 'Cannot find gt: {}'.format(name)
        assert name in pred_dict, 'Cannot find pred: {}'.format(name)
        assert os.path.exists(gt_dict[name]), 'Cannot find gt: {}'.format(gt_dict[name])
        assert os.path.exists(pred_dict[name]), 'Cannot find pred: {}'.format(pred_dict[name])

    pairs = [[gt_dict[name], pred_dict[name]] for name in names]
    info(logger, '{}\nIn total {} samples'.format(pred_root, len(pairs)))

    # iou
    iou = compute_iou(pairs, 81)
    info(logger, f'COCO mIoU: {iou.mean()}\n{iou}')
    return iou

def compute_iou(pairs, num_classes, map_func=None, num_threads=16, arr_=None):
    _compute_iou = lambda x: np.diag(x) / (x.sum(axis=0) + x.sum(axis=1) - np.diag(x) + 1e-10)

    if num_threads == 1:
        map_func = np.arange(max(num_classes, 256), dtype=np.int64) if map_func is None else map_func
        mat = np.zeros((num_classes, num_classes), np.float32)
        for gt_src, pred_src in pairs:
            gt = cv2.imread(gt_src, 0)
            pred = cv2.imread(pred_src, 0)
            if gt.shape != pred.shape:
                info(None, 'gt.shape != pred.shape. ({} vs. {}) for ({} vs. {})'.format(gt.shape, pred.shape, gt_src, pred_src))
                return mat
                pred = cv2.resize(pred, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
            
            gt = map_func[gt.ravel()]
            pred = map_func[pred.ravel()]
            #pred[pred >= num_classes] = 0
            valid = (gt < num_classes) & (pred < num_classes)
            mat += np.bincount(gt[valid] * num_classes + pred[valid], minlength=num_classes**2).reshape(num_classes, -1)

        if arr_ is not None:
            arr_mat = np.frombuffer(arr_.get_obj(), np.float32)
            arr_mat += mat.ravel()
        return _compute_iou(mat)
    else:
        workload = np.full((num_threads,), len(pairs) // num_threads, np.int64)
        if workload.sum() < len(pairs): workload[:len(pairs) - workload.sum()] += 1
        workload = np.cumsum(np.hstack([0, workload]))

        pairs_split = [pairs[i : j] for i, j in zip(workload[:-1], workload[1:])]
        arr_  = mp.Array('f', np.zeros((num_classes * num_classes,), np.float32))
        mat = np.frombuffer(arr_.get_obj(), np.float32).reshape(num_classes, -1)
        jobs = [mp.Process(target=compute_iou, args=(pairs_, num_classes, map_func, 1, arr_)) for pairs_ in pairs_split]
        [job.start() for job in jobs]
        [job.join() for job in jobs]
        return _compute_iou(mat.copy())

if __name__ == '__main__':
    gt_root = '/home/junsong_fan/diskf/data/VOC2012/extra/SegmentationClassAug'
    # data_list = './core/data/VOC/resources/train_aug.txt'

    # pred_root = './snapshot_cam/seed/irn_style'
    # pred_root = './snapshot/cian/vgg16_cian_Ep20_Bs32_Lr0.001_GPU4_Crop321_UseVFalse_shareQK/seed/irn_style'
    # pred_root = './snapshot/cian/vgg16_cian_Ep20_Bs32_Lr0.001_GPU4_Crop321_UseVFalse/seed/irn_style'
    # pred_root = './snapshot/cian/vgg16_cian_Ep20_Bs32_Lr0.001_GPU4_Crop321_UseVTrue/seed/irn_style'

    # for pred_root in [
    #         './snapshot/one/voc/vgg16_Ep20_Bs16_Lr0.001_GPU4_Size321x321_memonly_useLoss_T0.7_excW/seeds/train_aug',
    #         './snapshot/one/voc/vgg16_Ep20_Bs16_Lr0.001_GPU4_Size321x321_memonly_useLoss_T0.7_excW/seeds_crf/train_aug',
    #         './snapshot/one/voc/vgg16_Ep20_Bs16_Lr0.001_GPU4_Size321x321_memonly_useLoss_T0.7_excW/seeds_crf_conf/train_aug',
    #         './snapshot/one/voc/vgg16_Ep20_Bs16_Lr0.001_GPU4_Size321x321_memonly_useLoss_T0.7_excW/seeds_crf_noconf/train_aug'
    #         ]:
    #     compute_voc_iou(pred_root, gt_root, data_list)
    #compute_coco_iou(
    #        '../Data/Snapshots/MultiImage/coco/vgg16_largefov_T0.5_C64_mem_Lsim10.0/results/seeds/val',
    #        '../Data/Dataset/COCO/converted_labels/segmentation/val2014',
    #        'val'
    #)

    pred_root = '../Data/Snapshots/MCIS/mcis/r101_pami_revision/r101_aspp_T0.5_C64_All1e-4_EP20/results/seedsCRF/val'
    compute_voc_iou(pred_root, gt_root, 'val')
