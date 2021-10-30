from .vgg import *
from .caffe_resnet import *
from .attention_module import MultiheadAttention
from .memory import MemoryBank

def _resize_maybe(in_data, ref_data, mode='bilinear'):
    h1, w1 = in_data.shape[2:]
    h2, w2 = ref_data.shape[2:]
    if (h1 != h2) or (w1 != w2):
        in_data = F.interpolate(in_data, (h2, w2), mode=mode, align_corners=True)
    return in_data

class MCIS(nn.Module):
    def __init__(self,
        backboner,
        threshold,
        capacity,
        stop_attn_grad=True,
        use_ema=False,
        ema_momentum=None,
        mode='mcis',
        l2norm=False
    ):
        super(MCIS, self).__init__()
        self.backboner = backboner
        self.threshold = threshold
        self.capacity = capacity
        self.stop_attn_grad = stop_attn_grad
        self.use_ema = use_ema
        self.ema_momentum = ema_momentum
        self.l2norm = l2norm
        self.set_mode(mode)

        self.backbone = backboner()
        self.num_classes = self.backbone.num_classes

        num_heads = 1
        self.attention = MultiheadAttention(self.backbone.dim_feature, num_heads)
        self.attn_norm = nn.LayerNorm(self.backbone.dim_feature)

        self.queue = MemoryBank(
            self.num_classes,
            self.backbone.dim_feature,
            self.capacity * self.num_classes
        )

        self._make_ema_model()

    def _make_ema_model(self):
        if self.use_ema:
            self.backbone_ema = self.backboner()
            self.backbone_ema.eval()
            self._ema_param_initialized = False

    def _init_ema_params(self):
        assert not self._ema_param_initialized
        for p_ema, p in zip(self.backbone_ema.parameters(), self.backbone.parameters()):
            p_ema.requires_grad_(False)
            p_ema.data.copy_(p.data)
        self._ema_param_initialized = True

    def load_pretrained(self, pretrained, strict=True):
        self.backbone.load_pretrained(pretrained, strict)

    def get_param_groups(self):
        param_groups = self.backbone.get_param_groups()
        param_groups[10] = param_groups.get(10, []) + \
            list(self.attention.parameters()) + \
            list(self.attn_norm.parameters())
        
        # check the number of params
        nParams = len(list(self.parameters()))
        if self.use_ema:
            nParams -= len(list(self.backbone_ema.parameters()))
        nGroupParams = len(sum([list(v) for v in param_groups.values()], []))
        assert nParams == nGroupParams, (nParams, [len(v) for v in param_groups.values()])
        return param_groups

    def set_mode(self, mode):
        assert hasattr(self, f'forward_{mode}'), f'forward_{mode}'
        self.mode = mode
        self.forward = getattr(self, f'forward_{mode}')

    def forward_eval(self, image):
        feat = self.backbone.base(image)
        seg = self.backbone.head(feat)
        return seg

    @torch.no_grad()
    def forward_ema(self, image):
        assert self.use_ema
        if not self._ema_param_initialized:
            self._init_ema_params()
        if self.training:
            for p_ema, p in zip(self.backbone_ema.parameters(), self.backbone.parameters()):
                p_ema.data.copy_(p_ema.data * self.ema_momentum + p.data * (1 - self.ema_momentum))
        feat = self.backbone_ema.base(image)
        return feat

    @torch.no_grad()
    def get_enq_feat(self, feat, conf, label_cls):
        feat = feat.detach()

        conf = _resize_maybe(conf, feat)
        conf_maxima = conf.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        conf = torch.clamp_min(conf, 0) / torch.clamp_min(conf_maxima, 1e-5)
        conf = conf * label_cls.unsqueeze(2).unsqueeze(3)

        # weights for sum
        C = conf.size(1)
        thr_mask = (conf > self.threshold).float()
        max_mask = F.one_hot(conf.argmax(1), C).permute(0, 3, 1, 2).float()
        weight = thr_mask * max_mask * conf
        weight_sum = weight.sum((2, 3))

        # sum
        enq_feat = torch.bmm(weight.flatten(2), feat.flatten(2).transpose(1, 2))
        enq_feat = enq_feat / (weight_sum.unsqueeze(2) + 1e-5)
        enq_lbl = torch.arange(C, dtype=torch.int64, device=conf.device)
        enq_lbl = enq_lbl.unsqueeze(0).repeat(conf.size(0), 1)

        exist = weight_sum > 1
        enq_feat = enq_feat[exist].detach()
        enq_lbl = enq_lbl[exist].detach()
        return enq_feat, enq_lbl, weight

    @torch.no_grad()
    def get_enq_feat_pix(self, feat, conf, label_cls):
        feat = feat.detach()

        conf = _resize_maybe(conf, feat)
        conf_maxima = conf.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        conf = torch.clamp_min(conf, 0) / torch.clamp_min(conf_maxima, 1e-5)
        conf = conf * label_cls.unsqueeze(2).unsqueeze(3)

        C = conf.size(1)
        thr_mask = (conf > self.threshold).float()
        max_mask = F.one_hot(conf.argmax(1), C).permute(0, 3, 1, 2).float()
        mask_float = thr_mask * max_mask
        mask = mask_float.flatten(2).bool()

        N, C, S = mask.size()
        feat = feat.flatten(2).transpose(1, 2).contiguous()
        assert list(feat.shape[:2]) == [N, S], (feat.shape, mask.shape)

        feat_cands, feat_cands_lbl = [], []
        for i in range(N):
            for j in range(C):
                feat_cand = feat[i][mask[i, j]]
                if feat_cand.size(0) == 0:
                    continue
                feat_cand_lbl = torch.full((feat_cand.size(0),), j, dtype=torch.int64, device=feat.device)
                feat_cands.append(feat_cand)
                feat_cands_lbl.append(feat_cand_lbl)
        enq_feat = torch.cat(feat_cands, 0)
        enq_lbl = torch.cat(feat_cands_lbl, 0)
        return enq_feat, enq_lbl, mask_float


    @torch.no_grad()
    def get_deq_feat(self, label):
        label = label.detach()
        mem, mem_lbl = self.queue.pull(label.max(0)[0])
        m0, m1 = mem.size()
        mem = mem.unsqueeze(1).expand(m0, label.size(0), m1)
        mem_lbl_oh = F.one_hot(mem_lbl, label.size(1))
        mem_mask = (label.float() @ mem_lbl_oh.float().T) < 0.999
        return mem, mem_mask, mem_lbl

    def post_processing_attn(self, attn, size):
        attn = self.attn_norm(F.dropout(attn, p=0.1, training=self.training))
        attn = attn.permute(1, 2, 0).view(size).contiguous()
        attn = F.relu(attn, inplace=True)
        return attn

    def forward_mcis(self, image, label_cls):
        feat = self.backbone.base(image)
        cam = self.backbone.head_cam(feat)
        seg = self.backbone.head(feat)

        if self.stop_attn_grad:
            feat = feat.detach()
        feat_norm = F.normalize(feat) if self.l2norm else feat
        q = feat_norm.flatten(2).permute(2, 0, 1)

        if self.training:
            feat_ = self.forward_ema(image) if self.use_ema else feat
            feat_ = feat_.detach()
            feat_ = F.normalize(feat_) if self.l2norm else feat_
            #conf_ = torch.softmax(seg, 1).detach()
            conf_ = seg.detach()
            enq_feat, enq_label, enq_weight = self.get_enq_feat(feat_, conf_, label_cls)
            self.queue.push(enq_feat, enq_label)

        mem, mem_mask, mem_lbl = self.get_deq_feat(label_cls)
        attn, attn_w, attn_w_unnorm = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        attn = self.post_processing_attn(attn, feat.size())
        attn_feat = attn + feat
        attn_seg = self.backbone.head(attn_feat, stop_grad=self.stop_attn_grad)
        return cam, seg, attn_seg, enq_weight, attn_w_unnorm, mem_lbl

    def forward_mcis_eval(self, image, label_cls):
        feat = self.backbone.base(image)
        seg = self.backbone.head(feat)
        feat_norm = F.normalize(feat) if self.l2norm else feat
        q = feat_norm.flatten(2).permute(2, 0, 1)

        mem, mem_mask, mem_lbl = self.get_deq_feat(label_cls)
        attn, attn_w, attn_w_unnorm = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        attn = self.post_processing_attn(attn, feat.size())
        attn_feat = attn + feat
        attn_seg = self.backbone.head(attn_feat, stop_grad=self.stop_attn_grad)

        label_cls_2d = label_cls.unsqueeze(2).unsqueeze(3)
        seg = seg * label_cls_2d
        attn_seg = attn_seg * label_cls_2d
        return seg, attn_seg

    def forward_mcis_2stage(self, image, label_cls):
        feat = self.backbone.base(image)
        seg = self.backbone.head(feat)

        if self.stop_attn_grad:
            feat = feat.detach()
        feat_norm = F.normalize(feat) if self.l2norm else feat
        q = feat_norm.flatten(2).permute(2, 0, 1)

        if self.training:
            feat_ = self.forward_ema(image) if self.use_ema else feat
            feat_ = feat_.detach()
            feat_ = F.normalize(feat_) if self.l2norm else feat_
            #conf_ = torch.softmax(seg, 1).detach()
            conf_ = seg.detach()
            enq_feat, enq_label, enq_weight = self.get_enq_feat(feat_, conf_, label_cls)
            self.queue.push(enq_feat, enq_label)

        mem, mem_mask, mem_lbl = self.get_deq_feat(label_cls)
        attn, attn_w, attn_w_unnorm = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        attn = self.post_processing_attn(attn, feat.size())
        attn_feat = attn + feat
        attn_seg = self.backbone.head(attn_feat, stop_grad=self.stop_attn_grad)
        return seg, attn_seg, enq_weight, attn_w_unnorm, mem_lbl

    def forward_mcis_pix(self, image, label_cls):
        feat = self.backbone.base(image)
        cam = self.backbone.head_cam(feat)
        seg = self.backbone.head(feat)

        if self.stop_attn_grad:
            feat = feat.detach()
        feat_norm = F.normalize(feat) if self.l2norm else feat
        q = feat_norm.flatten(2).permute(2, 0, 1)

        if self.training:
            feat_ = self.forward_ema(image) if self.use_ema else feat
            feat_ = feat_.detach()
            feat_ = F.normalize(feat_) if self.l2norm else feat_
            conf_ = seg.detach()
            enq_feat, enq_label, enq_weight = self.get_enq_feat_pix(feat_, conf_, label_cls)
            self.queue.push(enq_feat, enq_label)

        mem, mem_mask, mem_lbl = self.get_deq_feat(label_cls)
        attn, attn_w, attn_w_unnorm = self.attention(q, mem, mem, key_padding_mask=mem_mask)
        attn = self.post_processing_attn(attn, feat.size())
        attn_feat = attn + feat
        attn_seg = self.backbone.head(attn_feat, stop_grad=self.stop_attn_grad)
        return cam, seg, attn_seg, enq_weight, attn_w_unnorm, mem_lbl



def get_backbone(model, num_classes):
    def wrapper():
        if model == 'vgg16_largefov':
            return VGG16_Sibling(num_classes, False)
        if model == 'vgg16_aspp':
            return VGG16_Sibling(num_classes, True)
        if model == 'r101_largefov':
            return Caffe_Resnet_Sibling(num_classes, [3, 4, 23, 3], False)
        if model == 'r101_aspp':
            return Caffe_Resnet_Sibling(num_classes, [3, 4, 23, 3], True)

        if model == 'plain_vgg16_largefov':
            return VGG16(num_classes, False)
        if model == 'plain_vgg16_aspp':
            return VGG16(num_classes, True)
        if model == 'plain_r101_largefov':
            return Caffe_Resnet(num_classes, [3, 4, 23, 3], False)
        if model == 'plain_r101_aspp':
            return Caffe_Resnet(num_classes, [3, 4, 23, 3], True)

        raise RuntimeError(model)
    return wrapper