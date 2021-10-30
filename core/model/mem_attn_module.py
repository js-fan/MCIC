import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_module import MultiheadAttention
from .memory import MemoryBank
from .vgg import VGG16_Sibling, VGG16_SelfAttn
from copy import deepcopy

class MemAttnModel(nn.Module):
    def __init__(self, backbone, capacity=64, threshold=0.7, num_heads=4, mode='train',
            mem_mode='reg', pull_mode='fix_class', use_attn_grad=False, add_self_attn=False,
            use_ema=False, ema_momentum=0.999, fix_length_mult=4):
        super(MemAttnModel, self).__init__()
        #assert isinstance(backbone, nn.Module)
        self.backbone = backbone()
        self.attention = MultiheadAttention(self.backbone.dim_feature, num_heads)
        self.attn_norm = nn.LayerNorm(self.backbone.dim_feature)

        self.capacity = capacity
        self.threshold = threshold
        self.num_heads = num_heads
        self.use_attn_grad = use_attn_grad
        self.add_self_attn = add_self_attn

        self.mode = mode
        self.mem_mode = mem_mode
        self.pull_mode = pull_mode
        self.fix_length_mult = fix_length_mult
        self.forward = getattr(self, 'forward_' + self.mode)
        self.fwk_mem = getattr(self, 'fwk_' + self.mem_mode)

        self.num_classes = self.backbone.num_classes
        self.queue = MemoryBank(self.num_classes, self.backbone.dim_feature,
                capacity=self.capacity*self.num_classes, mode=self.pull_mode,
                fix_length_mult=self.fix_length_mult)

        self.use_ema = use_ema
        self.ema_momentum = ema_momentum
        self.init_ema_model(backbone)

    def init_ema_model(self, backbone):
        self.backbone_ema = None
        if self.use_ema:
            self.backbone_ema = backbone()
            self.copy_ema_params()

    def copy_ema_params(self):
        for p_ema, p in zip(self.backbone_ema.parameters(), self.backbone.parameters()):
            p_ema.data.copy_(p.data)
            p_ema.requires_grad_(False)

    def get_param_groups(self):
        param_groups = self.backbone.get_param_groups()
        x10_params = list(self.attention.parameters()) + \
                list(self.attn_norm.parameters())
        param_groups.update({10: param_groups.get(10, []) + x10_params})
        return param_groups

    def load_pretrained(self, pretrained, strict=True):
        self.backbone.load_pretrained(pretrained, strict)
        if self.use_ema:
            self.copy_ema_params()

    def fwk_weight(self, feat, seg, label_cls):
        with torch.no_grad():
            h, w = feat.size()[2:]
            if seg.size(2) != h or seg.size(3) != w:
                seg = F.interpolate(seg, (h, w), mode='bilinear', align_corners=True)
            
            conf = seg - seg.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
            conf = conf / torch.clamp_min(conf.max(2, keepdim=True)[0].max(3, keepdim=True)[0], 1e-5)
            conf = conf * label_cls.unsqueeze(2).unsqueeze(3)
            mask = (conf > self.threshold).float()
            mask_oh = F.one_hot(conf.argmax(1), conf.size(1)).permute(0, 3, 1, 2).float().contiguous()
            mask = mask_oh * mask
        return mask, conf

    @torch.no_grad()
    def update_ema(self):
        for p_ema, p in zip(self.backbone_ema.parameters(), self.backbone.parameters()):
            p_ema *= self.ema_momentum
            p_ema += p * ( 1 - self.ema_momentum )
            #p_ema.data = p_ema.data * self.ema_momentum + p.data * (1 - self.ema_momentum)

    @torch.no_grad()
    def fwk_pix(self, feat, seg, label_cls):
        with torch.no_grad():
            mask, conf = self.fwk_weight(feat, seg, label_cls)
            selected = mask.max(1)[0] > 0.999
            feat = feat.permute(0, 2, 3, 1).contiguous()[selected].detach().clone()
            feat_lbl = mask.argmax(1)[selected].detach().clone()
        return feat, feat_lbl, mask

    @torch.no_grad()
    def fwk_reg(self, feat, seg, label_cls):
        with torch.no_grad():
            mask, conf = self.fwk_weight(feat, seg, label_cls)
            weight = mask * conf
            num_pix = weight.sum(axis=(2, 3))

            feat = torch.bmm(weight.flatten(2), feat.flatten(2).transpose(1, 2))
            feat = feat / torch.clamp_min(num_pix.unsqueeze(2), 1e-5)
            feat_lbl = torch.arange(seg.size(1), dtype=torch.int64, device=label_cls.device)
            feat_lbl = feat_lbl.unsqueeze(0).repeat(seg.size(0), 1)

            selected = label_cls.bool() & (num_pix > 0.999)
            feat = feat[selected].detach().clone()
            feat_lbl = feat_lbl[selected].detach().clone()
        return feat, feat_lbl, weight

    @torch.no_grad()
    def fwk_pull(self, label_cls):
        mem_mask = None
        with torch.no_grad():
            if self.pull_mode == 'fix_class':
                mem, mem_lbl = self.queue.pull(label_cls.max(0)[0])
                if mem is not None:
                    #mem = mem.unsqueeze(1).repeat(1, label_cls.size(0), 1)
                    m0, m1 = mem.size()
                    mem = mem.unsqueeze(1).expand(m0, label_cls.size(0), m1)
                    mem_lbl_oh = F.one_hot(mem_lbl, self.num_classes)
                    mem_mask = torch.matmul(label_cls.float(), mem_lbl_oh.float().T) < 0.999
            elif self.pull_mode == 'fix_length':
                mem, mem_lbl = self.queue.pull(label_cls)
            else:
                raise RuntimeError(self.pull_mode)
        return mem, mem_mask

    def forward_eval(self, image):
        feat, cls, seg = self.backbone(image)
        return seg

    def forward_cam(self, image):
        feat, cls, seg = self.backbone(image)
        return cls

    def forward_train(self, image, label_cls=None):
        feat, cls, seg = self.backbone(image)
        if label_cls is None:
            with torch.no_grad():
                label_cls_cls = (cls.mean(axis=(2, 3)) > 0).long()
                label_cls_cls = torch.cat([torch.ones((cls.size(0), 1), dtype=torch.int64, device=cls.device),
                    label_cls_cls], 1)
                label_cls_seg = F.one_hot(seg.argmax(1), self.num_classes).max(2)[0].max(1)[0]
                label_cls = label_cls_cls * label_cls_seg
        else:
            assert image.size(0) == label_cls.size(0), (image.size(), label_cls.size())

        m_weight = None
        if self.training:
            if self.use_ema:
                self.update_ema()
                with torch.no_grad():
                    feat_ema, cls_ema, seg_ema = self.backbone_ema(image)
            else:
                feat_ema = feat.detach()
                seg_ema = seg.detach()
            m_feat, m_feat_lbl, m_weight = self.fwk_mem(feat_ema, seg_ema, label_cls)
            self.queue.push(m_feat, m_feat_lbl)

        mem, mem_mask = self.fwk_pull(label_cls)
        if mem is None:
            return cls, seg, None, None


        if not self.use_attn_grad:
            feat = feat.detach()
        q = feat.flatten(2).permute(2, 0, 1)

        #mem_mask = None
        attn, _, _ = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        attn = self.attn_norm(F.dropout(attn, p=0.1, training=self.training))
        attn = attn.permute(1, 2, 0).view(feat.size()).contiguous() + feat

        if self.add_self_attn:
            attn_self, _, _ = self.attention(q, q, q)
            attn_self = self.attn_norm(F.dropout(attn_self, p=0.1, training=self.training))
            attn_self = attn_self.permute(1, 2, 0).view(feat.size()).contiguous()
            attn = attn + attn_self

        attn_seg = self.backbone.head(attn, not self.use_attn_grad)
        return cls, seg, attn_seg, m_weight

    def forward_no_attn(self, image):
        feat, cls, seg = self.backbone(image)
        return cls, seg

    def forward_self_attn(self, image):
        feat, cls, seg = self.backbone(image)

        if not self.use_attn_grad:
            feat = feat.detach()
        q = feat.flatten(2).permute(2, 0, 1)

        attn, _, _  = self.attention(q, q, q)
        attn = self.attn_norm(F.dropout(attn, p=0.1, training=self.training))
        attn = attn.permute(1, 2, 0).view(feat.size()).contiguous() + feat
        attn_seg = self.backbone.head(attn, not self.use_attn_grad)
        return cls, seg, attn_seg, None

    def forward_self_attn_merge(self, image):
        feat, cls, seg = self.backbone(image)

        if not self.use_attn_grad:
            feat = feat.detach()

    @torch.no_grad()
    def forward_exp_1_1(self, image):
        feat, cls, seg = self.backbone(image)
        q = feat.flatten(2).permute(2, 0, 1)

        attn, weight, weight_logit = self.attention(q, q, q)
        attn = self.attn_norm(F.dropout(attn, p=0.1, training=self.training))
        attn = attn.permute(1, 2, 0).view(feat.size()).contiguous() + feat
        attn_seg = self.backbone.head(attn, not self.use_attn_grad)

        #prob = F.softmax(seg, 1)
        #prob_attn = F.softmax(attn_seg, 1)
        #b, c, h, w= seg.size()
        #weight = weight.view(b, h, w, h, w)
        return seg, attn_seg, weight


def get_model(model, num_classes, *args, **kwargs):
    assert model in ['vgg16_largefov', 'vgg16_aspp', 'vgg16_selfattn', 
            'resnet101_largefov', 'resnet101_aspp'], model
    def backbone_creater():
        if model == 'vgg16_largefov':
            return VGG16_Sibling(num_classes, use_aspp=False)
        if model == 'vgg16_aspp':
            return VGG16_Sibling(num_classes, use_aspp=True)
        if model == 'vgg16_selfattn':
            return VGG16_SelfAttn(num_classes, use_aspp=False)
        raise NotImplementedError

    # if model == 'vgg16_largefov':
    #     backbone = VGG16_Sibling(num_classes, use_aspp=False)
    # elif model == 'vgg16_aspp':
    #     backbone = VGG16_Sibling(num_classes, use_aspp=True)
    # elif model == 'vgg16_selfattn':
    #     backbone = VGG16_SelfAttn(num_classes, use_aspp=False)
    # else:
    #     raise NotImplementedError
    return MemAttnModel(backbone_creater, *args, **kwargs)

