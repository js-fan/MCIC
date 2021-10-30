import torch
import numpy as np
import cv2
import os
from ...utils import meani_list
_curr_path = os.path.dirname(os.path.abspath(__file__))

PADDING_VALUE = 0

def CHECK_EXISTS(src):
    assert os.path.exists(src), src

class COCOVersatileDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, split, target_size,
        segment_root=None, superpixel_root=None, max_superpixel=1024,
        annotation_root=None, image_label_use_background=False,
        rand_crop=False, rand_mirror=False, rand_scale=None, return_src=False,
        downsample_label=1, repeat=1
    ):
        name_file = os.path.join(_curr_path, 'resources', split+'.txt')
        assert os.path.exists(name_file), name_file
        with open(name_file) as f:
            srcs = [x.strip().split(' ')[0] for x in f.readlines()]
        names = [x.rsplit('.', 1)[0] for x in srcs]

        # image srcs
        self.data_images = [os.path.join(image_root, x) for x in srcs]
        for src in self.data_images:
            CHECK_EXISTS(src)

        # segmentation maps (opt.)
        if segment_root is not None:
            if isinstance(segment_root, str):
                segment_root = [segment_root]
            self.data_segments = [[os.path.join(root, x+'.png') for root in segment_root] for x in names]

            for src_list in self.data_segments:
                for src in src_list:
                    CHECK_EXISTS(src)
        else:
            self.data_segments = None
        
        # superpixels (opt.)
        if superpixel_root is not None:
            raise NotImplementedError
        else:
            self.data_superpixels = None

        # image-level labels
        if annotation_root is not None:
            with open(annotation_root) as f:
                anns = [x.strip().split(' ') for x in f.readlines()]
            anns = {x[0]: np.array([int(L) for L in x[1:]]) for x in anns}
            self.data_labels = [anns[x] for x in srcs]
            if image_label_use_background:
                self.data_labels = [x + 1 for x in self.data_labels]
        else:
            self.data_labels = None

        # attrs
        self.target_size = target_size
        self.rand_crop = rand_crop
        self.rand_mirror = rand_mirror
        self.rand_scale = rand_scale
        self.return_src = return_src
        self.downsample_label = downsample_label
        self.max_superpixel = max_superpixel
        self.image_label_use_background = image_label_use_background
        self.repeat = repeat

    def __getitem__(self, index):
        src = self.data_images[index]
        lbl = self.data_labels[index] if self.data_labels is not None else None
        seg_srcs = self.data_segments[index] if self.data_segments is not None else None
        sp_src = self.data_superpixels[index] if self.data_superpixels is not None else None

        # img, segs, sp
        return_vals = fn_load_transform(src, seg_srcs, sp_src,
                self.target_size, self.rand_crop, self.rand_mirror, self.rand_scale,
                self.downsample_label, self.max_superpixel
        )

        # img, lbl, segs, sp
        if lbl is not None:
            image_label = torch.zeros((81 if self.image_label_use_background else 80,), dtype=torch.int64)
            for L in lbl:
                image_label[L] = 1
            return_vals.insert(1, image_label)

        # img, lbl, segs, sp, src
        if self.return_src:
            return_vals.append(src)

        return_vals = [x for x in return_vals if x is not None]

        for _ in range(self.repeat-1):
            repeats = fn_load_transform(src, seg_srcs, sp_src,
                    self.target_size, self.rand_crop, self.rand_mirror, self.rand_scale,
                    self.downsample_label, self.max_superpixel
            )
            return_vals += [x for x in repeats if x is not None]
        return return_vals

    def __len__(self):
        return len(self.data_images)

def fn_load_transform(src, seg_srcs, sp_src,
        target_size, rand_crop, rand_mirror, rand_scale,
        downsample_label, max_superpixel):
    # load data
    img = cv2.imread(src)
    h, w = img.shape[:2]

    has_seg = seg_srcs is not None
    if has_seg:
        segs = [cv2.imread(x, 0) for x in seg_srcs]
        assert all([x.shape == (h, w) for x in segs]), [x.shape for x in segs] + [img.shape]
    else:
        segs = [None]

    has_sp = sp_src is not None
    if has_sp:
        sp = cv2.imread(sp_src).astype(np.int64)
        assert sp.shape[:2] == (h, w), [sp.shape, img.shape]
        sp = sp[..., 0] + sp[..., 1] * 256 + sp[..., 2] * 65536
    else:
        sp = None

    # rand scale
    if rand_scale is not None:
        scale = np.random.uniform(rand_scale[0], rand_scale[1])
        #scale = np.random.choice([0.5, 0.75, 1, 1.25, 1.5])
        h = int(h * scale + .5)
        w = int(w * scale + .5)
        img = cv2.resize(img, (w, h))

        if has_seg: segs = [cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in segs]
        if has_sp: sp = cv2.resize(sp, (w, h), interpolation=cv2.INTER_NEAREST)

    # rand mirror
    if rand_mirror and np.random.rand() > 0.5:
        img = img[:, ::-1]
        if has_seg: segs = [x[:, ::-1] for x in segs]
        if has_sp: sp = sp[:, ::-1]

    # rand crop
    if target_size is not None:
        ph, pw = max(target_size[0] - h, 0), max(target_size[1] - w, 0)
        pt, pl = ph // 2, pw // 2
        pb, pr = ph - pt, pw - pl
        if ph > 0 or pw > 0:
            img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=meani_list)
            if has_seg: segs = [cv2.copyMakeBorder(x, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=PADDING_VALUE) for x in segs]
            if has_sp: sp = cv2.copyMakeBorder(sp, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=int(sp.max())+1)
        h, w = img.shape[:2]

        if rand_crop:
            bh = np.random.randint(0, h - target_size[0] + 1)
            bw = np.random.randint(0, w - target_size[1] + 1)
        else:
            bh = (h - target_size[0]) // 2
            bw = (w - target_size[1]) // 2

        img = img[bh : bh+target_size[0], bw : bw + target_size[1]]
        if has_seg: segs = [x[bh : bh+target_size[0], bw : bw + target_size[1]] for x in segs]
        if has_sp: sp = sp[bh : bh+target_size[0], bw : bw + target_size[1]]

    # to torch
    img = img[..., ::-1].transpose(2, 0, 1).astype(np.float32) - np.array(meani_list[::-1], dtype=np.float32).reshape(3, 1, 1)
    img = torch.from_numpy(img)
    ih, iw = img.size()[1:]

    if has_seg:
        if downsample_label != 1:
            dh, dw = (ih // downsample_label) + (ih % 2), (iw // downsample_label) + (iw % 2)
            segs = [cv2.resize(x, (dw, dh), interpolation=cv2.INTER_NEAREST) for x in segs]
        segs = [torch.from_numpy(x.astype(np.int64)) for x in segs]

    if has_sp:
        if downsample_label != 1:
            dh, dw = (ih // downsample_label) + (ih % 2), (iw // downsample_label) + (iw % 2)
            sp = cv2.resize(sp, (dw, dh), interpolation=cv2.INTER_NEAREST)
        # check if |sp| < max_superpixel
        sp_ids, sp2, sp_areas = np.unique(sp, return_counts=True, return_inverse=True)
        if len(sp_ids) > max_superpixel:
            to_keep = sp_areas.argsort()[-max_superpixel:]
            lookup = np.full((sp_ids.size,), max_superpixel, np.int64)
            lookup[to_keep] = np.arange(max_superpixel)
            sp2 = lookup[sp2]
        sp = sp2.reshape(sp.shape)
        sp = torch.from_numpy(sp)

    return [img] + segs + [sp]

