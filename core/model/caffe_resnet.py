import torch
import torch.nn as nn
import torch.nn.functional as F
import os

_BATCH_NORM = nn.BatchNorm2d
_BOTTLENECK_EXPANSION = 4

class _ConvBNReLU(nn.Sequential):
    BATCH_NORM = _BATCH_NORM
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True):
        super(_ConvBNReLU, self).__init__()
        self.add_module('conv', nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False))
        self.add_module('bn', _BATCH_NORM(out_ch, eps=1e-5, momentum=1-0.999))
        if relu:
            self.add_module('relu', nn.ReLU())

class _Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBNReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBNReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBNReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = _ConvBNReLU(in_ch, out_ch, 1, stride, 0, 1, False) if downsample else nn.Identity()
    
    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)

class _ResLayer(nn.Sequential):
    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()
        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        for i in range(n_layers):
            self.add_module(f'block{i+1}', _Bottleneck(
                in_ch if i == 0 else out_ch,
                out_ch,
                stride if i == 0 else 1,
                dilation * multi_grids[i],
                True if i == 0 else False
            ))

class _Stem(nn.Sequential):
    def __init__(self, out_ch):
        super(_Stem, self).__init__()
        self.add_module('conv1', _ConvBNReLU(3, out_ch, 7, 2, 3, 1, True))
        self.add_module('pool', nn.MaxPool2d(3, 2, 1, ceil_mode=True))
    
class _ResLargeFov(nn.Module):
    def __init__(self, in_ch, num_classes, dilation=12, drop=False):
        super(_ResLargeFov, self).__init__()
        self.fc1 = nn.Conv2d(in_ch, num_classes, 3, 1, dilation, dilation)
        self.drop = drop
    
    def forward(self, x, stop_grad=False):
        if stop_grad:
            prev_state = self.fc1.weight.requires_grad
            self.fc1.requires_grad_(False)
        
        if self.drop:
            x = F.dropout(x, p=0.5, training=self.training)
        out = self.fc1(x)

        if stop_grad:
            self.fc1.requires_grad_(prev_state)
        return out

    def get_x10_params(self):
        return [self.fc1.weight, self.fc1.bias]

class _ResASPP(nn.Module):
    def __init__(self, inplanes, num_classes, dilation=[6, 12, 18, 24], drop=False):
        super(_ResASPP, self).__init__()
        self.aspp0 = _ResLargeFov(inplanes, num_classes, dilation[0], drop=drop)
        self.aspp1 = _ResLargeFov(inplanes, num_classes, dilation[1], drop=drop)
        self.aspp2 = _ResLargeFov(inplanes, num_classes, dilation[2], drop=drop)
        self.aspp3 = _ResLargeFov(inplanes, num_classes, dilation[3], drop=drop)

    def forward(self, x, stop_grad=False):
        x0 = self.aspp0(x, stop_grad)
        x1 = self.aspp1(x, stop_grad)
        x2 = self.aspp2(x, stop_grad)
        x3 = self.aspp3(x, stop_grad)
        out = x0 + x1 + x2 + x3
        return out

    def get_x10_params(self):
        return sum([getattr(self, f'aspp{i}').get_x10_params() for i in range(4)], [])

class _Resnet_Base(nn.Module):
    def __init__(self, n_blocks):
        super(_Resnet_Base, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.layer1 = _Stem(ch[0])
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1)
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1)
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2)
        self.layer5 = _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4)

    def forward(self, x, as_dict=False):
        c0 = self.layer1(x)
        c1 = self.layer2(c0)
        c2 = self.layer3(c1)
        c3 = self.layer4(c2)
        c4 = self.layer5(c3)
        if as_dict:
            return {f'c{i}': eval(f'c{i}') for i in range(5)}
        return c4

class Caffe_Resnet(nn.Module):
    def __init__(self, num_classes, n_blocks, use_aspp=False, largefov_dilation=12):
        super(Caffe_Resnet, self).__init__()
        self.base = _Resnet_Base(n_blocks)

        ch = [64 * 2 ** p for p in range(6)]
        if use_aspp:
            self.head = _ResASPP(ch[5], num_classes)
        else:
            self.head = _ResLargeFov(ch[5], num_classes, largefov_dilation)

        self.num_classes = num_classes
        self.use_aspp = use_aspp
        self.dim_feature = ch[5]

    def forward(self, x):
        c4 = self.base(x)
        out = self.head(c4)
        return out

    def convert_init_params(self, data):
        out_data = {}
        for k, v in data.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            out_data[k] = v

        for i in range(4):
            for suffix in ['weight', 'bias']:
                del out_data[f'base.aspp.c{i}.{suffix}']

        if self.use_aspp:
            for i in range(4):
                out_data[f'head.aspp{i}.fc1.weight'] = torch.zeros((self.num_classes, self.dim_feature, 3, 3), dtype=torch.float32)
                out_data[f'head.aspp{i}.fc1.bias'] = torch.zeros((self.num_classes,), dtype=torch.float32)
                nn.init.normal_(out_data[f'head.aspp{i}.fc1.weight'], 0, 0.01)
        else:
            out_data['head.fc1.weight'] = torch.zeros((self.num_classes, self.dim_feature, 3, 3), dtype=torch.float32)
            out_data['head.fc1.bias'] = torch.zeros((self.num_classes,), dtype=torch.float32)
            nn.init.normal_(out_data['head.fc1.weight'], 0, 0.01)
        return out_data

    def load_pretrained(self, data, strict=True):
        if isinstance(data, str):
            assert os.path.exists(data), data
            data = torch.load(data, map_location='cpu')
        data = self.convert_init_params(data) 
        #device = list(self.parameters())[0].device
        #check_data = data[f"head{'.aspp0' if self.use_aspp else ''}.fc1.weight"]
        #print(device, check_data.data.cpu().numpy().ravel()[:10])
        self.load_state_dict(data, strict=strict)

    def get_x10_params(self):
        return self.head.get_x10_params()

    def get_param_groups(self):
        x10_params = self.get_x10_params()
        x1_params = []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if all([p is not p10 for p10 in x10_params]):
                x1_params.append(p)
        return {1: x1_params, 10: x10_params}

class Caffe_Resnet_Sibling(Caffe_Resnet):
    def __init__(self, num_classes, n_blocks, use_aspp=False, largefov_dilation=12, mode='cam_seg'):
        super(Caffe_Resnet_Sibling, self).__init__(num_classes, n_blocks, use_aspp, largefov_dilation)
        self.head_cam = _ResLargeFov(self.dim_feature, self.num_classes-1, 1)
        self.set_mode(mode)

    def forward_cam_seg(self, x):
        c5 = self.base(x)
        cam = self.head_cam(c5)
        seg = self.head(c5)
        return cam, seg

    def forward_eval(self, x):
        return super(Caffe_Resnet_Sibling, self).forward(x)

    def set_mode(self, mode):
        assert hasattr(self, f'forward_{mode}')
        self.forward = getattr(self, f'forward_{mode}')
        self.mode = mode

    def convert_init_params(self, data):
        out_data = super(Caffe_Resnet_Sibling, self).convert_init_params(data)
        out_data['head_cam.fc1.weight'] = torch.zeros((self.num_classes-1, self.dim_feature, 3, 3), dtype=torch.float32)
        out_data['head_cam.fc1.bias'] = torch.zeros((self.num_classes-1,), dtype=torch.float32)
        nn.init.normal_(out_data['head_cam.fc1.weight'], 0, 0.01)
        return out_data

    def get_x10_params(self):
        return self.head.get_x10_params() + self.head_cam.get_x10_params()

if __name__ == '__main__':
    mod = Caffe_Resnet_Sibling(21, [3, 4, 23, 3], True)
    x = torch.rand(1, 3, 321, 321).float()
    outs = mod(x)
    print([y.shape for y in outs])
