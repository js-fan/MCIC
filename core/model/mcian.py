from numpy.lib.index_tricks import nd_grid
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .attention_module import MultiheadAttention
from .memory import MemoryBank
from .vgg import VGG16_Sibling, VGG16_SelfAttn, LargeFov
from .resnet_deeplab import DeeplabV2_Sibling, DeeplabV2_Sibling_C3
from .caffe_resnet import Caffe_Resnet_Sibling


# method also utilizes self-attn
class MCIAN(nn.Module):
    def __init__(self, backbone, mode=None, block_cam_grad=False, block_attn_grad=False,
        threshold=None, capacity=None, use_ema=False, ema_momentum=None, use_separate_attn=False, pixmem_max_num=None
    ):
        super(MCIAN, self).__init__()
        self.backbone = backbone(block_cam_grad=block_cam_grad)
        num_heads = 4
        self.attention = MultiheadAttention(self.backbone.dim_feature, num_heads)
        self.attn_norm = nn.LayerNorm(self.backbone.dim_feature)
        self.block_attn_grad = block_attn_grad

        self.num_classes = self.backbone.num_classes
        self.threshold = threshold
        self.capacity = capacity
        self.queue = None
        if self.threshold is not None:
            assert self.capacity is not None
            self.queue = MemoryBank(self.num_classes, self.backbone.dim_feature,
            capacity=self.capacity)

        self.set_mode(mode)

        self.use_ema = use_ema
        self.ema_momentum = ema_momentum
        self.make_ema_model(backbone)

        self.use_separate_attn = use_separate_attn
        if self.use_separate_attn:
            self.attention2 = MultiheadAttention(self.backbone.dim_feature, num_heads)
        
        self.pixmem_max_num = pixmem_max_num

    def make_ema_model(self, backbone):
        self.backbone_ema = None
        if self.use_ema:
            assert self.ema_momentum is not None
            self.backbone_ema = backbone()
            self.backbone_ema.eval()
            self._ema_param_initialized = False

    def init_ema_params(self):
        assert not self._ema_param_initialized
        for p_ema, p in zip(self.backbone_ema.parameters(), self.backbone.parameters()):
            p_ema.requires_grad_(False)
            p_ema.data.copy_(p.data)
        self._ema_param_initialized = True

    def set_mode(self, mode):
        self.mode = 'base' if mode is None else mode
        assert hasattr(self, 'forward_' + self.mode), "Cannot find 'forward_{}'.".format(self.mode)
        self.forward = getattr(self, 'forward_' + self.mode)
        if self.mode == 'mem':
            assert self.threshold is not None
            assert self.queue is not None

    def load_pretrained(self, pretrained, strict=True):
        self.backbone.load_pretrained(pretrained, strict)

    def get_param_groups(self):
        param_groups = self.backbone.get_param_groups()
        x10_params = list(self.attention.parameters()) + \
                list(self.attn_norm.parameters())
        param_groups.update({10: param_groups.get(10, []) + x10_params})

        # make sure that params are all present here
        nParams = len(list(self.parameters()))
        nGroupParams = len(sum([list(v) for v in param_groups.values()], []))
        if self.use_ema:
            nParams -= len(list(self.backbone_ema.parameters()))
        assert nParams == nGroupParams, (nParams, [len(v) for v in param_groups.values()])

        return param_groups

    def forward_base(self, image):
        feat, cls, seg = self.backbone(image)
        return feat, cls, seg

    @torch.no_grad()
    def forward_ema(self, image):
        assert self.use_ema
        if not self._ema_param_initialized:
            self.init_ema_params()
        if self.training:
            for p_ema, p in zip(self.backbone_ema.parameters(), self.backbone.parameters()):
                p_ema.data.copy_(p_ema.data * self.ema_momentum + p.data * (1 - self.ema_momentum))
        feat, cls, seg = self.backbone_ema(image)
        return feat, cls, seg

    def forward_self(self, image):
        feat, cls, seg = self.backbone(image)

        if self.block_attn_grad:
            feat = feat.detach()
        q = feat.flatten(2).permute(2, 0, 1)
        attn, attn_weights, attn_weights_unnorm = self.attention(q, q, q)
        attn = self.post_process_attn(attn, feat.size()) + feat
        attn_seg = self.backbone.head(attn, self.block_attn_grad)
        return cls, seg, attn_seg, attn_weights

    def forward_self_eval(self, image):
        cls, seg, attn_seg, attn_weights = self.forward_self(image)
        return attn_seg
    
    def forward_cam(self, image):
        feat = self.backbone.forward_base(image)
        cls = self.backbone.head_cam(feat)
        return cls

    def forward_mem(self, image, label_cls):
        feat, cls, seg = self.backbone(image)

        if self.block_attn_grad:
            feat = feat.detach()
        q = feat.flatten(2).permute(2, 0, 1)

        # get attn k, v
        eq_weight = None
        if self.training:
            if self.use_ema:
                feat_, _, seg_ = self.forward_ema(image)
            else:
                feat_ = feat
            feat_eq, feat_eq_lbl, eq_weight = self.get_enqueue_feat(feat_, seg, label_cls)
            self.queue.push(feat_eq, feat_eq_lbl)

        if label_cls is None:
            assert not self.training
            label_cls = (cls.detach().mean(axis=(2, 3)) > 0).long()
            label_cls = torch.cat([
                torch.ones((label_cls.size(0), 1), dtype=torch.int64, device=label_cls.device),
                label_cls], 1)
        mem, mem_mask, mem_lbl = self.get_dequeue_feat(label_cls)
        if mem is None:
            return cls, seg, None, None, None
        
        attn, attn_weights, attn_weights_unnorm = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        attn = self.post_process_attn(attn, feat.size()) + feat
        attn_seg = self.backbone.head(attn, block_grad=self.block_attn_grad)
        return cls, seg, attn_seg, eq_weight, [attn_weights_unnorm, mem_lbl]
    
    def forward_pixmem(self, image, label_cls):
        feat, cls, seg = self.backbone(image)

        if self.block_attn_grad:
            feat = feat.detach()
        q = feat.flatten(2).permute(2, 0, 1)

        # get attn k, v
        eq_weight = None
        if self.training:
            if self.use_ema:
                feat_, _, seg_ = self.forward_ema(image)
            else:
                feat_ = feat
            feat_eq, feat_eq_lbl, eq_weight = self.get_enqueue_feat_pix(feat_, seg, label_cls, self.pixmem_max_num)
            self.queue.push(feat_eq, feat_eq_lbl)

        if label_cls is None:
            assert not self.training
            label_cls = (cls.detach().mean(axis=(2, 3)) > 0).long()
            label_cls = torch.cat([
                torch.ones((label_cls.size(0), 1), dtype=torch.int64, device=label_cls.device),
                label_cls], 1)
        mem, mem_mask, mem_lbl = self.get_dequeue_feat(label_cls)
        if mem is None:
            return cls, seg, None, None, None
        
        attn, attn_weights, attn_weights_unnorm = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        attn = self.post_process_attn(attn, feat.size()) + feat
        attn_seg = self.backbone.head(attn, block_grad=self.block_attn_grad)
        return cls, seg, attn_seg, eq_weight, [attn_weights_unnorm, mem_lbl]
    
    def forward_debug(self, image, label_cls):
        feat, cls, seg = self.backbone(image)
        q = feat.flatten(2).permute(2, 0, 1)
        feat_eq, feat_eq_lbl, eq_weight = self.get_enqueue_feat(feat, seg, label_cls)
        self.queue.push(feat_eq, feat_eq_lbl)
        mem, mem_mask, mem_lbl = self.get_dequeue_feat(label_cls, self.length_limit)
        attn, attn_weights, attn_weights_unnorm = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        attn = self.post_process_attn(attn, feat.size()) + feat
        attn_seg = self.backbone.head(attn, self.block_attn_grad)
        return cls, seg, attn_seg, eq_weight, attn_weights

    def forward_mem_cam(self, image, label_cls):
        assert label_cls is not None
        feat, cls, seg = self.backbone(image)
        if self.block_attn_grad:
            feat = feat.detach()
        q = feat.flatten(2).permute(2, 0, 1)

        # get attn k, v
        eq_weight = None
        if self.training:
            if self.use_ema:
                feat_, _, seg_ = self.forward_ema(image)
            else:
                feat_ = feat
            feat_eq, feat_eq_lbl, eq_weight = self.get_enqueue_feat(feat_, seg, label_cls)
            self.queue.push(feat_eq, feat_eq_lbl)
        mem, mem_mask, mem_lbl = self.get_dequeue_feat(label_cls)
        if mem is None:
            return cls, seg, None, None

        attn, attn_weights, attn_weights_unnorm = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        attn = self.post_process_attn(attn, feat.size()) + feat
        attn_cam = self.backbone.head_cam(attn, self.block_attn_grad)
        attn_seg = self.backbone.head(attn, self.block_attn_grad)
        return cls + attn_cam, seg, attn_seg, eq_weight
    
    def forward_mem_cam_only(self, image, label_cls):
        assert label_cls is not None
        feat, cls, seg = self.backbone(image)
        if self.block_attn_grad:
            feat = feat.detach()
        q = feat.flatten(2).permute(2, 0, 1)

        # get attn k, v
        eq_weight = None
        if self.training:
            if self.use_ema:
                feat_, _, seg_ = self.forward_ema(image)
            else:
                feat_ = feat
            feat_eq, feat_eq_lbl, eq_weight = self.get_enqueue_feat(feat_, seg, label_cls)
            self.queue.push(feat_eq, feat_eq_lbl)
        mem, mem_mask, mem_lbl = self.get_dequeue_feat(label_cls)
        if mem is None:
            return cls, seg, None, None

        attn, attn_weights, attn_weights_unnorm = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        attn = self.post_process_attn(attn, feat.size()) + feat
        attn_cam = self.backbone.head_cam(attn, self.block_attn_grad)
        return cls + attn_cam, seg, None, None

    def forward_mem_eval(self, image, label_cls=None):
        feat, cls, seg = self.backbone(image)
        if label_cls is not None:
            pass
        return seg
    
    def forward_baseline(self, image, label_cls=None):
        feat, cls, seg = self.backbone(image)
        return cls, seg, None, None, None

    def forward_retrain(self, image):
        feat = self.backbone.forward_base(image)
        seg = self.backbone.head(feat)
        return seg

    def forward_mem_inbatch(self, image, label_cls):
        feat, cls, seg = self.backbone(image)

        if self.block_attn_grad:
            feat = feat.detach()
        q = feat.flatten(2).permute(2, 0, 1)

        # get attn k, v
        mem, mem_lbl, eq_weight = self.get_enqueue_feat(feat, seg, label_cls)
        m0, m1 = mem.size()
        mem = mem.unsqueeze(1).expand(m0, label_cls.size(0), m1)
        mem_lbl_oh = F.one_hot(mem_lbl, self.num_classes)
        mem_mask = torch.matmul(label_cls.float(), mem_lbl_oh.float().T) <  0.999

        attn, attn_weights, attn_weights_unnorm = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        attn = self.post_process_attn(attn, feat.size()) + feat
        attn_seg = self.backbone.head(attn, self.block_attn_grad)
        return cls, seg, attn_seg, eq_weight

    def forward_mem_insample(self, image, label_cls):
        feat, cls, seg = self.backbone(image)

        if self.block_attn_grad:
            feat = feat.detach()
        q = feat.flatten(2).permute(2, 0, 1)

        # get attn k, v
        mem, mem_lbl, eq_weight, selected = self.get_enqueue_feat(feat, seg, label_cls, True)
        m0, m1 = mem.size()
        mem = mem.unsqueeze(1).expand(m0, label_cls.size(0), m1)
        selected_index = torch.arange(seg.size(0), dtype=torch.int64, device=seg.device)
        selected_index = selected_index.unsqueeze(1).repeat(1, seg.size(1))
        selected_index = selected_index[selected]
        selected_index_oh = F.one_hot(selected_index, seg.size(0))
        mem_mask = selected_index_oh.T < 0.999

        attn, attn_weights, attn_weights_unnorm = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        attn = self.post_process_attn(attn, feat.size()) + feat
        attn_seg = self.backbone.head(attn, self.block_attn_grad)
        return cls, seg, attn_seg, eq_weight

    def forward_mem_self(self, image, label_cls):
        feat, cls, seg = self.backbone(image)

        assert self.block_attn_grad
        if self.block_attn_grad:
            feat = feat.detach()
        q = feat.flatten(2).permute(2, 0, 1)

        attn, attn_weights, attn_weights_unnorm = self.attention(q, q, q)
        attn = self.post_process_attn(attn, feat.size()) + feat
        attn_seg = self.backbone.head(attn, self.block_attn_grad)
        return cls, seg, attn_seg, attn_weights

    def get_norm_tensor4d(self, conf):
        # conf = conf - conf.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
        conf = torch.clamp_min(conf, 0)
        conf = conf / torch.clamp_min(conf.max(2, keepdim=True)[0].max(3, keepdim=True)[0], 1e-5)
        return conf

    @torch.no_grad()
    def get_enqueue_feat(self, feat, seg, label_cls, return_index=False):
        feat = feat.detach()
        label_cls = label_cls.detach()

        h, w = feat.size()[2:]
        if seg.size(2) != h or seg.size(3) != w:
            seg = F.interpolate(seg, (h, w), mode='bilinear', align_corners=True)
        
        conf = self.get_norm_tensor4d(seg) * label_cls.unsqueeze(2).unsqueeze(3)
        mask = (conf > self.threshold).float()
        #mask_oh = F.one_hot(conf.argmax(1), conf.size(1)).permute(0, 3, 1, 2).float().contiguous()
        mask_oh = F.one_hot((F.softmax(seg, 1) * label_cls.unsqueeze(2).unsqueeze(3)).argmax(1), conf.size(1)).permute(0, 3, 1, 2).float().contiguous()
        mask = mask_oh * mask
        weight = mask * conf
        num_pix = weight.sum(axis=(2, 3))

        feat = torch.bmm(weight.flatten(2), feat.flatten(2).transpose(1, 2))
        feat = feat / torch.clamp_min(num_pix.unsqueeze(2), 1e-5)
        feat_lbl = torch.arange(seg.size(1), dtype=torch.int64, device=seg.device)
        feat_lbl = feat_lbl.unsqueeze(0).repeat(seg.size(0), 1)

        selected = label_cls.bool() & (num_pix >= 1)
        feat_eq = feat[selected].detach().clone()
        feat_eq_lbl = feat_lbl[selected].detach().clone()
        if return_index:
            return feat_eq, feat_eq_lbl, weight, selected.detach()
        return feat_eq, feat_eq_lbl, weight

    @torch.no_grad()
    def get_dequeue_feat(self, label_cls):
        label_cls = label_cls.detach()
        mem_mask = None
        mem, mem_lbl = self.queue.pull(label_cls.max(0)[0])
        if mem is not None:
            m0, m1 = mem.size()
            mem = mem.unsqueeze(1).expand(m0, label_cls.size(0), m1)
            mem_lbl_oh = F.one_hot(mem_lbl, self.num_classes)
            mem_mask = torch.matmul(label_cls.float(), mem_lbl_oh.float().T) < 0.999
        return mem, mem_mask, mem_lbl
    
    @torch.no_grad()
    def get_enqueue_feat_pix(self, feat, seg, label_cls, max_num):
        feat = feat.detach()
        label_cls = label_cls.detach()

        h, w = feat.size()[2:]
        if seg.size(2) != h or seg.size(3) != w:
            seg = F.interpolate(seg, (h, w), mode='bilinear', align_corners=True)
        
        conf = self.get_norm_tensor4d(seg) * label_cls.unsqueeze(2).unsqueeze(3)
        mask = (conf > self.threshold).float()
        mask_oh = F.one_hot((F.softmax(seg, 1) * label_cls.unsqueeze(2).unsqueeze(3)).argmax(1), conf.size(1)).permute(0, 3, 1, 2).float().contiguous()
        mask_float = mask_oh * mask
        mask = mask_float.flatten(2).bool()

        n, c, s = mask.size()
        feat = feat.flatten(2).transpose(1, 2).contiguous()
        assert list(feat.size()[:2]) == [n, s], (feat.size(), mask.size())

        feat_cands, feat_cands_lbl = [], []
        for i in range(n):
            for j in range(c):
                feat_cand = feat[i][mask[i, j]]
                if feat_cand.size(0) == 0:
                    continue
                if (max_num is not None) and (feat_cand.size(0) > max_num):
                    index = torch.randperm(feat_cand.size(0))[:max_num]
                    feat_cand = feat_cand[index]
                feat_cand_lbl = torch.full((feat_cand.size(0),), j, dtype=torch.int64, device=feat.device)
                feat_cands.append(feat_cand)
                feat_cands_lbl.append(feat_cand_lbl)
        feat_eq = torch.cat(feat_cands, dim=0)
        feat_eq_lbl = torch.cat(feat_cands_lbl, dim=0)
        return feat_eq, feat_eq_lbl, mask_float
        
    def forward_mem_useself(self, image, label_cls):
        feat, cls, seg = self.backbone(image)

        if self.block_attn_grad:
            feat = feat.detach()
        q = feat.flatten(2).permute(2, 0, 1)

        # self attention
        if self.use_separate_attn:
            attn_s, attn_weights_s, attn_weights_unnorm_s = self.attention2(q, q, q)
        else:
            attn_s, attn_weights_s, attn_weights_unnorm_s = self.attention(q, q, q)

        # get weights that will be multiplied on seg as conf
        n, dim, h, w = feat.size()
        with torch.no_grad():
            spatial_weights = attn_weights_s
            spatial_weights = spatial_weights.max(1)[0].view(n, 1, h, w)
            spatial_weights = self.get_norm_tensor4d(spatial_weights)
        # spatial_weights = 1

        # get attn k, v
        eq_weight = None
        if self.training:
            if self.use_ema:
                feat_, _, seg_ = self.forward_ema(image)
            else:
                feat_ = feat
            feat_eq, feat_eq_lbl, eq_weight = self.get_enqueue_feat(feat_, seg * spatial_weights, label_cls)
            self.queue.push(feat_eq, feat_eq_lbl)

        assert label_cls is not None
        mem, mem_mask, mem_lbl = self.get_dequeue_feat(label_cls)
        if mem is None:
            return cls, seg, None, None

        attn_c, attn_weights_c, attn_weights_unnorm_c = self.attention(q, mem, mem, key_padding_mask=mem_mask)

        # use attn_s and attn_c
        PRE_MERGE = False
        if PRE_MERGE:
            attn = self.post_process_attn(attn_s + attn_c, feat.size())
        else:
            attn = self.post_process_attn(attn_s, feat.size()) + \
                self.post_process_attn(attn_c, feat.size())
        # attn = self.post_process_attn(attn_c, feat.size())
        attn = attn + feat
        attn_seg = self.backbone.head(attn, self.block_attn_grad)
        return cls, seg, attn_seg, eq_weight
    
    def post_process_attn(self, attn, size):
        attn = self.attn_norm(F.dropout(attn, p=0.1, training=self.training))
        attn = attn.permute(1, 2, 0).view(size).contiguous()
        attn = F.relu(attn, inplace=True)
        return attn

    @torch.no_grad()
    def forward_visdemo_th(self, image, label_cls):
        feat, cls, seg = self.backbone(image)

        conf = self.get_norm_tensor4d(seg) * label_cls.unsqueeze(2).unsqueeze(3)
        mask_oh = F.one_hot((F.softmax(seg, 1) * label_cls.unsqueeze(2).unsqueeze(3)).argmax(1), conf.size(1)).permute(0, 3, 1, 2).float().contiguous()
        out_weights = []
        for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
            mask = (conf > th).float() * mask_oh
            weight = mask * conf
            out_weights.append(weight)
        return out_weights

    @torch.no_grad()
    def forward_visdemo_ent(self, image, label_cls):
        feat, cls, seg = self.backbone(image)
        q = feat.flatten(2).permute(2, 0, 1)
        #mem, mem_lbl = self.queue.pull_all()
        mem, mem_mask, mem_lbl = self.get_dequeue_feat(label_cls)
        assert mem is not None
        attn, attn_weights, attn_weights_unnorm = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        return feat, attn_weights_unnorm, mem_lbl
        
class MCIAN_Sepseg(MCIAN):
    def __init__(self, *args, **kwargs):
        super(MCIAN_Sepseg, self).__init__(*args, **kwargs)
        self.attn_seg = LargeFov(self.backbone.dim_feature, self.num_classes)

    def load_pretrained(self, pretrained, strict=True):
        data = self.backbone.load_pretrained(pretrained, strict)
        attn_seg_data = {k.replace('head.', ''): v for k, v in data.items() if k.startswith('head.')}
        self.attn_seg.load_state_dict(attn_seg_data, strict=strict)
        return data.update(attn_seg_data)

    def get_param_groups(self):
        param_groups = super(MCIAN_Sepseg, self).get_param_groups()
        x10_params = [self.attn_seg.fc3.weight, self.attn_seg.fc3.bias]
        x1_params = [self.attn_seg.fc1.weight, self.attn_seg.fc1.bias,
            self.attn_seg.fc2.weight, self.attn_seg.fc2.bias
        ]
        param_groups.update({10: param_groups.get(10, []) + x10_params, 
            1: param_groups.get(1, []) + x1_params
        })
        nParams = len(list(self.parameters()))
        nGroupParams = len(sum([list(v) for v in param_groups.values()], []))
        assert nParams == nGroupParams, (nParams, [len(v) for v in param_groups])
        return param_groups

    def forward_mem(self, image, label_cls):
        feat, cls, seg = self.backbone(image)

        if self.block_attn_grad:
            feat = feat.detach()
        q = feat.flatten(2).permute(2, 0, 1)

        eq_weight = None
        if self.training:
            if self.use_ema:
                feat_, _, seg_ = self.forward_ema(image)
            else:
                feat_ = feat
            feat_eq, feat_eq_lbl, eq_weight = self.get_enqueue_feat(feat_, seg, label_cls)
            self.queue.push(feat_eq, feat_eq_lbl)
        
        mem, mem_mask, mem_lbl = self.get_dequeue_feat(label_cls)
        if mem is None:
            return cls, seg, None, None, None
        
        attn, attn_weights, attn_weights_unnorm = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        attn = self.post_process_attn(attn, feat.size()) + feat
        attn_seg = self.attn_seg(attn)
        return cls, seg, attn_seg, eq_weight, [attn_weights_unnorm, mem_lbl]


def get_model(model, num_classes, *args, **kwargs):
    def backbone_creater(*_args, **_kwargs):
        if model == 'vgg16_largefov':
            return VGG16_Sibling(num_classes, False, *_args, **_kwargs)
        if model == 'vgg16_aspp':
            return VGG16_Sibling(num_classes, True, *_args, **_kwargs)
        if model == 'vgg16_selfattn':
            return VGG16_SelfAttn(num_classes, False, *_args, **_kwargs)
        if model == 'vgg16_largefov_union':
            return VGG16_Union(num_classes, False, *_args, **_kwargs)
        if model == 'vgg16_aspp_union':
            return VGG16_Union(num_classes, True, *_args, **_kwargs)
        if model == 'r101_largefov':
            return Caffe_Resnet_Sibling(num_classes, [3, 4, 23, 3], False, *_args, **_kwargs)
        if model == 'r101_aspp':
            return Caffe_Resnet_Sibling(num_classes, [3, 4, 23, 3], True, *_args, **_kwargs)
        '''
        if model == 'r101_largefov':
            return DeeplabV2_Sibling([3, 4, 23, 3], num_classes, False, *_args, **_kwargs)
        if model == 'r50_largefov':
            return DeeplabV2_Sibling([3, 4, 6, 3], num_classes, False, *_args, **_kwargs)
        if model == 'r101_aspp':
            return DeeplabV2_Sibling([3, 4, 23, 3], num_classes, True, *_args, **_kwargs)
        if model == 'r101_aspp_msc':
            return DeeplabV2_Sibling([3, 4, 23, 3], num_classes, True, True, *_args, **_kwargs)
        if model == 'r50_aspp':
            return DeeplabV2_Sibling([3, 4, 6, 3], num_classes, True, *_args, **_kwargs)
        if model == 'r101_c3_largefov':
            return DeeplabV2_Sibling_C3([3, 4, 6, 3], num_classes, False, *_args, **_kwargs)
        if model == 'r101_c3_aspp':
            return DeeplabV2_Sibling_C3([3, 4, 23, 3], num_classes, True, *_args, **_kwargs)
        if model == 'vgg16_largefov_sepseg':
            return VGG16_Sibling(num_classes, False, *_args, **_kwargs)
        '''
        raise RuntimeError(model)

    if model.endswith('_sepseg'):
        return MCIAN_Sepseg(backbone_creater, *args, **kwargs)
    else:
        return MCIAN(backbone_creater, *args, **kwargs)
