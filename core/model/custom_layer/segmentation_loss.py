import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logit, label):
        assert logit.dim() == 4, logit.size()
        assert label.dim() == 3, label.size()

        softmax = F.softmax(logit, 1)
        n, c, h, w = softmax.size()

        label_oh = F.one_hot(label, 256)[..., :c].permute(0, 3, 1, 2).float().contiguous()
        if (label_oh.size(2) != h) or (label_oh.size(3) != w):
            label_oh = F.interpolate(label_oh, (h, w), mode='bilinear')
            val, idx = label_oh.max(1)
            label_oh = F.one_hot(idx, c).permute(0, 3, 1, 2).float().contiguous() * \
                    (val.unsqueeze(1) > 0.5).float()

        label_mask = label_oh.max(1, keepdim=True)[0]
        loss = - ( label_oh * torch.log(softmax + 1e-5) ).sum() / (label_mask.sum() + 1e-5)

        #lr_mult = torch.ones((1,), device=loss.device, dtype=loss.dtype)[0] * lr_mult
        #ctx.save_for_backward(softmax, label_oh, label_mask, lr_mult)
        ctx.save_for_backward(softmax, label_oh, label_mask)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        #softmax, label_oh, label_mask, lr_mult = ctx.saved_tensors
        softmax, label_oh, label_mask = ctx.saved_tensors
        grad_logit = grad_label = None

        if ctx.needs_input_grad[0]:
            num_pix = torch.clamp_min(label_mask.sum(), 1)
            grad_logit = ( (softmax - label_oh) * label_mask ) * (grad_output / num_pix)

        return grad_logit, grad_label

segmentation_loss = SegmentationLoss.apply
