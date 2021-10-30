import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os

def _make_conv_block(num_conv, num_filter_in, num_filter_out, pool=None, dilation=1, inplace=True, drop=None, kernel=3):
    pad = (kernel // 2) if (dilation == 1) else dilation
    layers = OrderedDict()

    for i in range(num_conv):
        _num_filter_in = num_filter_in if i == 0 else num_filter_out
        layers['{}'.format(i)] = nn.Conv2d(_num_filter_in, num_filter_out, kernel, 1, pad, dilation, bias=True)
        layers['{}_relu'.format(i)] = nn.ReLU(inplace=inplace)
    if pool:
        layers['pool'] = nn.MaxPool2d(3, pool, 1)
    if drop:
        layers['drop'] = nn.Dropout(drop)
    return nn.Sequential(layers)

class _VGG16_Largefov(nn.Module):
    def __init__(self, inplanes, num_classes, dilation=12, drop=True):
        super(_VGG16_Largefov, self).__init__()
        self.fc1 = nn.Conv2d(inplanes, 1024, 3, 1, dilation, dilation)
        self.fc2 = nn.Conv2d(1024, 1024, 1)
        self.fc3 = nn.Conv2d(1024, num_classes, 1)
        self.drop = drop

    def forward(self, x, stop_grad=False):
        if stop_grad:
            prev_state = self.fc1.weight.requires_grad
            self.fc1.requires_grad_(False)
            self.fc2.requires_grad_(False)
            self.fc3.requires_grad_(False)

        x = F.relu(self.fc1(x), inplace=True)
        if self.drop:
            x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x), inplace=True)
        if self.drop:
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)

        if stop_grad:
            self.fc1.requires_grad_(prev_state)
            self.fc2.requires_grad_(prev_state)
            self.fc3.requires_grad_(prev_state)
        return x

    def get_x10_params(self):
        return [self.fc3.weight, self.fc3.bias]

class _VGG16_ASPP(nn.Module):
    def __init__(self, inplanes, num_classes, dilation=[6, 12, 18, 24], drop=True):
        super(_VGG16_ASPP, self).__init__()
        self.aspp0 = _VGG16_Largefov(inplanes, num_classes, dilation[0], drop=drop)
        self.aspp1 = _VGG16_Largefov(inplanes, num_classes, dilation[1], drop=drop)
        self.aspp2 = _VGG16_Largefov(inplanes, num_classes, dilation[2], drop=drop)
        self.aspp3 = _VGG16_Largefov(inplanes, num_classes, dilation[3], drop=drop)

    def forward(self, x, stop_grad=False):
        x0 = self.aspp0(x, stop_grad)
        x1 = self.aspp1(x, stop_grad)
        x2 = self.aspp2(x, stop_grad)
        x3 = self.aspp3(x, stop_grad)
        out = x0 + x1 + x2 + x3
        return out

    def get_x10_params(self):
        return sum([getattr(self, f'aspp{i}').get_x10_params() for i in range(4)], [])

class _VGG16_Base(nn.Module):
    def __init__(self):
        super(_VGG16_Base, self).__init__()
        self.conv1 = _make_conv_block(2,   3,  64, 2)
        self.conv2 = _make_conv_block(2,  64, 128, 2)
        self.conv3 = _make_conv_block(3, 128, 256, 2)
        self.conv4 = _make_conv_block(3, 256, 512, 1)
        self.conv5 = _make_conv_block(3, 512, 512, 1, 2)
        self.avg_pool = nn.AvgPool2d(3, 1, 1)

    def forward(self, x, as_dict=False):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c5 = self.avg_pool(c5)
        if as_dict:
            return {f'c{i}': eval(f'c{i}') for i in range(1, 6)}
        return c5

class VGG16(nn.Module):
    def __init__(self, num_classes, use_aspp=False, largefov_dilation=12):
        super(VGG16, self).__init__()
        self.base = _VGG16_Base()
        if use_aspp:
            self.head = _VGG16_ASPP(512, num_classes)
        else:
            self.head = _VGG16_Largefov(512, num_classes, largefov_dilation)
        self.num_classes = num_classes
        self.use_aspp = use_aspp
        self.dim_feature = 512

    def forward(self, x):
        c5 = self.base(x)
        out = self.head(c5)
        return out

    def convert_init_params(self, data):
        out_data = {}
        for k, v in data.items():
            if k.startswith('module.'):
                k = k[len('module.'):]

            if k.startswith('conv'):
                out_data[f'base.{k}'] = v
            elif k.startswith('fc'):
                layer = int(k[2]) - 5
                tgt_names = [f'head.aspp{i}.fc{layer}' for i in range(4)] if self.use_aspp else [f'head.fc{layer}']
                suffix = k.rsplit('.', 1)[1]
                for tgt_name in tgt_names:
                    if layer == 3:
                        if suffix == 'weight':
                            v = torch.zeros((self.num_classes, 1024, 1, 1), dtype=torch.float32)
                            nn.init.normal_(v, 0, 0.01)
                        else:
                            v = torch.zeros((self.num_classes,), dtype=torch.float32)
                    out_data[f'{tgt_name}.{suffix}'] = v.clone()
        return out_data

    def load_pretrained(self, data, strict=True):
        if isinstance(data, str):
            assert os.path.exists(data), data
            data = torch.load(data, map_location='cpu')
        data = self.convert_init_params(data)
        self.load_state_dict(data, strict=strict)
        return data

    def get_x10_params(self):
        return self.head.get_x10_params()

    def get_param_groups(self):
        x10_params = self.get_x10_params()
        x1_params = []
        for p in self.parameters():
            if all([p is not p10 for p10 in x10_params]):
                x1_params.append(p)
        return {1: x1_params, 10: x10_params}

class VGG16_Sibling(VGG16):
    def __init__(self, num_classes, use_aspp=False, largefov_dilation=12, mode='cam_seg'):
        super(VGG16_Sibling, self).__init__(num_classes, use_aspp, largefov_dilation)
        self.head_cam = _VGG16_Largefov(512, self.num_classes-1, 1)
        self.set_mode(mode)

    def forward_cam_seg(self, x):
        c5 = self.base(x)
        cam = self.head_cam(c5)
        seg = self.head(c5)
        return cam, seg

    def forward_eval(self, x):
        return super(VGG16_Sibling, self).forward(x)

    def set_mode(self, mode):
        assert hasattr(self, f'forward_{mode}')
        self.forward = getattr(self, f'forward_{mode}')
        self.mode = mode

    def convert_init_params(self, data):
        out_data = super(VGG16_Sibling, self).convert_init_params(data)
        for i in range(1, 3):
            src = f'head.aspp0.fc{i}' if self.use_aspp else f'head.fc{i}'
            tgt = f'head_cam.fc{i}'
            out_data[f'{tgt}.weight'] = out_data[f'{src}.weight'].clone()
            out_data[f'{tgt}.bias'] = out_data[f'{src}.bias'].clone()

        out_data['head_cam.fc3.weight'] = torch.zeros((self.num_classes-1, 1024, 1, 1), dtype=torch.float32)
        out_data['head_cam.fc3.bias'] = torch.zeros((self.num_classes-1,), dtype=torch.float32)
        nn.init.normal_(out_data['head_cam.fc3.weight'], 0, 0.01)
        return out_data

    def get_x10_params(self):
        return self.head.get_x10_params() + self.head_cam.get_x10_params()


