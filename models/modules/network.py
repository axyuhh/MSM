import functools
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.spectral_norm  import spectral_norm
from .cbin import CBINorm2d
from .adain import AdaINorm2d
from torchvision.models import vgg19

###########################################################################
#                                                                         #
#                            external network                             #
#                                                                         #
###########################################################################
def get_decoder():
    return Generator()

def get_style_encoder():
    return StyleEncoder()

def get_discriminator():
    return Discriminator()

def weights_init(init_type='xavier'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif (classname.find('Norm') == 0):
            if hasattr(m, 'weight') and m.weight is not None:
                init.constant(m.weight.data, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
    return init_fun

class ResidualBlock(nn.Module):
    def __init__(self, h_dim, norm_layer=None, nl_layer=None, use_dropout=False):
        super(ResidualBlock, self).__init__()
        self.use_dropout = use_dropout
        self.block1 = conv3x3(h_dim,h_dim,norm_layer=norm_layer,nl_layer=nl_layer)
        self.block2 = conv3x3(h_dim,h_dim,norm_layer=norm_layer)
        self.block3 = nn.Dropout(0.5)

    def forward(self, x):
        y = self.block1(x) + x
        y = self.block2(y)
        if self.use_dropout:
            y = self.block3(y)
        return x+y

class ConResidualBlock(nn.Module):
    def __init__(self, h_dim, c_norm_layer=None, nl_layer=None, use_dropout=False, return_con=False):
        super(ConResidualBlock, self).__init__()
        self.c1 = Conv2dBlock(h_dim,h_dim, kernel_size=3, stride=1, padding=1,pad_type='reflect', bias=False)
        self.n1 = c_norm_layer(h_dim)
        self.l1 = nl_layer()
        self.c2 = Conv2dBlock(h_dim,h_dim, kernel_size=3, stride=1, padding=1, pad_type='reflect', bias=False)
        self.n2 = c_norm_layer(h_dim)
    def forward(self, x, code):
        y = self.l1(self.n1(self.c1(x), code))
        y = self.n2(self.c2(y), code)
        out = x + y
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out

class Upsampling2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=None, nl_layer=None, c_norm_layer=None, norm_normal=None):
        super(Upsampling2dBlock, self).__init__()
        self.firstcon = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.last_norm = norm_normal(in_dim)

        self.SAB = SpatialAttention()
        self.CAB = ChannelAttention(in_dim)

        self.cheng_con = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.jia_con = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)

        self.source_conv = Conv2dBlock(in_dim, in_dim, kernel_size=3, stride=1, padding=1,
                    pad_type='reflect', bias=False, norm_layer=norm_normal, nl_layer=nl_layer)

        self.cheng_sty = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.jia_sty = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)

        self.n1 = c_norm_layer(in_dim)
        self.l1 = nl_layer()

        self.n2 = c_norm_layer(in_dim)
        self.l2 = nl_layer()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv2dBlock(in_dim,out_dim,kernel_size=3, stride=1, padding=1,
                pad_type='reflect', bias=False,norm_layer=norm_layer,nl_layer=nl_layer)
            )
        self.last_acti = nl_layer()

    def forward(self, x, content, code):
        norm_input = self.last_norm(self.firstcon(x))

        content = content * self.SAB(content) * self.CAB(content)
        content_cheng = self.cheng_con(content) + 1.0
        content_jia = self.jia_con(content)

        x = self.source_conv(x)
        x = x * content_cheng + content_jia

        style_cheng = self.l1(self.n1(self.cheng_sty(x), code)) + 1.0
        style_jia = self.l2(self.n2(self.jia_sty(x), code))

        last = norm_input * style_cheng + style_jia
        last = self.last_acti(last)

        return self.upsample(last)

class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape),
                                self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)

class Generator(nn.Module):
    def __init__(self, nef=64, input_nc=256, n_latent=8, n_ds_blocks=2, n_resblocks=2):
        super().__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type='adain', num_con=n_latent)
        nl_layer = get_nl_layer(layer_type='lrelu')
        self.block1 = nn.ModuleList([Conv2dBlock(1, nef, kernel_size=7, stride=1, padding=3,
                                                pad_type='reflect', bias=False, norm_layer=norm_layer,
                                                nl_layer=nl_layer)])
        input_nef = nef
        for _ in range(0, n_ds_blocks):
            output_nef = min(input_nef * 2, 512)
            self.block1.append(Conv2dBlock(input_nef, output_nef, kernel_size=4, stride=2, padding=1,
                                          pad_type='reflect', bias=False, norm_layer=norm_layer, nl_layer=nl_layer))
            input_nef = output_nef
        for _ in range(n_resblocks):
            self.block1.append(ResidualBlock(input_nef, norm_layer=norm_layer, nl_layer=nl_layer))

        self.block2 = nn.ModuleList([])
        for i in range(n_resblocks):
            self.block2.append(ResidualBlock(input_nc, norm_layer=norm_layer, nl_layer=nl_layer))
        for i in range(n_ds_blocks):
            self.block2.append(Upsampling2dBlock(input_nc, input_nc // 2, norm_layer=LayerNorm, nl_layer=nl_layer, c_norm_layer=c_norm_layer, norm_normal=norm_layer))
            input_nc = input_nc // 2
        self.block2 += [Conv2dBlock(input_nc, 1, kernel_size=7, stride=1, padding=3,
                                   pad_type='reflect', bias=True, nl_layer=nn.Tanh)]

    def forward(self, x, s):
        output = []
        num = 0
        for box in self.block1:
            num = num + 1
            x = box(x)
            if num == 2 or num == 3:
                output.append(x) # 0 1
        num = 0
        for box in self.block2:
            num = num + 1
            if num == 1:
                out = box(x)
            elif num == 2:
                out = box(out)
            elif num == 3:
                out = box(out, output[1], s)
            elif num == 4:
                out = box(out, output[0], s)
            elif num == 5:
                out = box(out)
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0,
                 pad_type='reflect', bias=True, norm_layer=None, nl_layer=None):
        super(Conv2dBlock, self).__init__()
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        self.conv = spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size,
                                            stride=stride, padding=0, bias=bias))
        if norm_layer is not None:
            self.norm = norm_layer(out_dim)
        else:
            self.norm = lambda x: x

        if nl_layer is not None:
            self.activation = nl_layer()
        else:
            self.activation = lambda x: x

    def forward(self, x):
        return self.activation(self.norm(self.conv(self.pad(x))))

def conv3x3(in_dim, out_dim, norm_layer=None, nl_layer=None):
    return Conv2dBlock(in_dim, out_dim, kernel_size=3, stride=1, padding=1,
            pad_type='reflect', bias=False, norm_layer=norm_layer, nl_layer=nl_layer)

class DownResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=None, nl_layer=None):
        super(DownResidualBlock, self).__init__()
        self.encode = nn.Sequential(
                        norm_layer(in_dim),
                        nl_layer(),
                        conv3x3(in_dim, in_dim,norm_layer=norm_layer,nl_layer=nl_layer),
                        conv3x3(in_dim, out_dim),
                        nn.AvgPool2d(kernel_size=2, stride=2))
        self.shortcut = nn.Sequential(
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Conv2dBlock(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True))
    def forward(self, x):
        y = self.encode(x)
        y = y + self.shortcut(x)
        return y

def get_norm_layer(layer_type='cbin', num_con=0):
    if layer_type == 'cbin':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        c_norm_layer = functools.partial(CBINorm2d, affine=False, num_con=num_con)
    elif layer_type == 'adain':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        c_norm_layer = functools.partial(AdaINorm2d, num_con=num_con)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer, c_norm_layer

def get_nl_layer(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'sigmoid':
        nl_layer = nn.Sigmoid
    elif layer_type == 'tanh':
        nl_layer = nn.Tanh
    else:
        raise NotImplementedError('nl_layer layer [%s] is not found' % layer_type)
    return nl_layer

class StyleEncoder(nn.Module):
    def __init__(self, max_conv_dim=256):
        super().__init__()
        dim_in = 64
        norm_layer = None
        nl_layer = get_nl_layer(layer_type='relu')
        blocks = []
        blocks += [Conv2dBlock(1, dim_in, kernel_size=7, stride=1, padding=3,
                                                norm_layer=norm_layer,
                                                nl_layer=nl_layer,
                                                pad_type='reflect')]
        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [Conv2dBlock(dim_in, dim_out, kernel_size=4, stride=2, padding=1,
                                                norm_layer=norm_layer,
                                                nl_layer=nl_layer,
                                                pad_type='reflect')]
            dim_in = dim_out

        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.Conv2d(dim_out, 8, 1, 1, 0)]
        self.shared = nn.Sequential(*blocks)

    def forward(self, x):
        s = self.shared(x).view(x.size(0), -1)
        return s

class ConDownResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, c_norm_layer=None, nl_layer=None, return_con=False):
        super(ConDownResidualBlock, self).__init__()
        self.return_con = return_con
        self.cnorm1 = c_norm_layer(in_dim)
        self.nl1 = nl_layer()
        self.conv1 = conv3x3(in_dim, in_dim)
        self.cnorm2 = c_norm_layer(in_dim)
        self.nl2 = nl_layer()
        self.cmp = nn.Sequential(
                        conv3x3(in_dim, out_dim),
                        nn.AvgPool2d(kernel_size=2, stride=2))
        self.shortcut = nn.Sequential(
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Conv2dBlock(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        out = self.cmp(self.nl2(self.cnorm2(self.conv1(self.nl1(self.cnorm1(x))))))
        out = out + self.shortcut(x)
        return out

def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_norm_layerD(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

norm_layer = get_norm_layerD(norm_type='instance')

class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=norm_layer, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
