class MCIAN_Sepseg(MCIAN):
    def __init__(self, *args, **kwargs):
        super(MCIAN_Sepseg, self).__init__(*args, **kwargs)
        self.attn_seg = LargeFov(self.backbone.dim_feature, self.num_classes)

    def load_pretrained(self, pretrained, strict=True):
        data = self.backbone.load_pretrained(pretrained, strict)
        attn_seg_data = {k.replace('head.', 'attn_seg.'): v for k, v in data.items() if k.startswith('head.')}
        self.attn_seg.load_state_dict(attn_seg_data, strict=strict)
        return data.update(attn_seg_data)

    def get_param_groups(self):
        param_groups = super(MCIAN_Sepseg, self).get_param_groups()
        x10_params = [self.attn_seg.fc3.weight, self.attn_seg.fc3.bias]
        param_groups.update({10: param_groups.get(10, []) + x10_params})
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
        attn_seg = self.attn_head(attn)
        return cls, seg, attn_seg, eq_weight, [attn_weights_unnorm, mem_lbl]


