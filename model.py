import gc
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, nglobal, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.globalfeature = nn.Sequential( # For DCSNN
            nn.Linear(512 * block.expansion, nglobal*2),
            nn.ReLU(True),
            nn.Linear(nglobal*2, nglobal)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.globalfeature(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, nglobal, block, layers, pretrained, progress, **kwargs):
    model = ResNet(nglobal, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
    return model


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def simple_nms(scores, nms_radius: int): # TODO: understand this routine
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def sample_descriptors(keypoints, descriptors, s: int = 8): 
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s/2 + 0.5 # TODO: interpolation coordinates check
    keypoints = keypoints/torch.tensor([(w*s - s), (h*s - s)],).to(keypoints)[None]
    # keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    # TODO : keypoints in x,y? or h,w?
    # https://discuss.pytorch.org/t/solved-torch-grid-sample/51662

    descriptors = torch.nn.functional.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


from typing import Callable, Any, Optional, List
class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class LocalFeatureGenerator(nn.Module):
    def __init__(self, base_model, nlocal, n_class = 1):
        super().__init__()
        self.nms_radius = 2
        self.keypoint_threshold = 0.005
        self.remove_borders = 4
        self.max_keypoints = 800

        self.base_model = base_model
        self.base_layers = list(self.base_model.children())

        self.down1 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.down2 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)   
        self.down3 = self.base_layers[5]
        self.down4 = self.base_layers[6]
        self.down5 = self.base_layers[7]

        self.up1 = Up(512+256, 256)
        # self.up2 = Up(256+128, 256)
        self.up2 = Up(256+128, 128)
        self.up3 = Up(128+64, 64)
        self.up4 = Up(64+64, 64)
        self.up5 = Up(64, 64)

        # self.kp_outconv_a = nn.Conv2d(256, 128, kernel_size = 1)
        # self.kp_outconv_b = nn.Conv2d(128, 65, kernel_size = 1)
        # self.ld_outconv = nn.Conv2d(256, self.localdesc_dim, kernel_size = 1)
        
        self.kp_outconv = nn.Conv2d(64, 1, kernel_size = 1)
        # self.ld_outconv = nn.Conv2d(64, nlocal, kernel_size = 3, padding=1)
        self.ld_outconv = nn.Conv2d(64, nlocal, kernel_size = 1, padding=1)
        # self.test = InvertedResidual(64, 256, 1, 1)

    def forward(self, input):
        x1 = self.down1(input) # bs, 64, 112, 112
        x2 = self.down2(x1) # bs, 64, 56, 56
        x3 = self.down3(x2) # bs, 128, 28, 28
        x4 = self.down4(x3) # bs, 256, 14, 14
        x5 = self.down5(x4) # bs, 512, 7, 7
        x = self.up1(x5,x4) # bs, 256, 14, 14
        x = self.up2(x,x3) # bs, 128, 28, 28
        
        x = self.up3(x,x2) # bs, 64, 56, 56
        x = self.up4(x,x1) # bs, 64, 112, 112
        x = self.up5(x)
        
        kpts_map = torch.sigmoid(self.kp_outconv(x))
        kpts_map_nms = simple_nms(kpts_map, self.nms_radius).squeeze(1)
        kpts_idxs = [torch.nonzero(kpts, as_tuple=False) for kpts in kpts_map_nms]

        kpts_idxs, kpts_score = list(zip(*[top_k_keypoints(kp_idx, kp_val[kp_idx[:,0],kp_idx[:,1]], self.max_keypoints) for kp_idx, kp_val in zip(kpts_idxs, kpts_map_nms)]))
        
        if x.requires_grad:
            x.retain_grad()

        localdesc = self.ld_outconv(x)
        localdesc = F.normalize(localdesc, p=2, dim=1) # normalize localdesc
        localdesc_kp = [ld[:, kp_idx[:,0], kp_idx[:,1]] for ld, kp_idx in zip(localdesc, kpts_idxs)]
        
        # aa = [kpidx.size() for kpidx in kpts_idxs]
        # print(aa)

        return kpts_idxs, localdesc_kp, kpts_score, kpts_map.squeeze()

        # # keypoint output
        # scores = F.relu(self.kp_outconv_a(x))
        # scores = self.kp_outconv_b(scores)
        # scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        # b, _, h, w = scores.shape
        # scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        # scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        # scores = simple_nms(scores, self.nms_radius)
        # scores_max = scores.flatten(1).max(axis=1).values
        # scores_normalized = torch.stack([s/sm for sm, s in zip(scores_max, scores)])

        # # Extract keypoints
        # keypoints = [torch.nonzero(s > self.keypoint_threshold, as_tuple=False) for s in scores]

        # # Extract keypoints map
        # # keypoints_map = torch.stack([torch.where(s > self.keypoint_threshold, scores.new_tensor(1), scores.new_tensor(0)) for s in scores])

        # scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)] # Score for each keypoint

        # # Discard keypoints near the image borders
        # keypoints, scores = list(zip(*[remove_borders(k, s, self.remove_borders, h*8, w*8) for k, s in zip(keypoints, scores)]))

        # # Keep the k keypoints with highest score
        # if self.max_keypoints >= 0:
        #     keypoints, scores = list(zip(*[top_k_keypoints(k, s, self.max_keypoints) for k, s in zip(keypoints, scores)]))

        # # Convert (h, w) to (x, y)
        # keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # # kp_out = self.kp_outconv(x) # bs, 1, 112, 112
        # localdesc = self.ld_outconv(x)
        # localdesc = torch.nn.functional.normalize(localdesc, p=2, dim=1)
        
        # # Interpolate descriptors
        # localdesc = [sample_descriptors(k[None], d[None], 8)[0] for k, d in zip(keypoints, localdesc)]

        # return keypoints, localdesc, scores, scores_normalized

        
def dcsnn_full(nglobal, nlocal, pretrained = True, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    encoder = _resnet('resnet18', nglobal, BasicBlock, [2, 2, 2, 2], pretrained, progress)
    encoder_k = _resnet('resnet18', nglobal, BasicBlock, [2, 2, 2, 2], pretrained, progress)
    lf_generator = LocalFeatureGenerator(encoder, nlocal)
    net = DCSNN_Full(lf_generator, encoder_k, dim=nglobal, **kwargs)
    return net


class DCSNN_Full(nn.Module):
    def __init__(self, lf_generator, encoder_k, **kwargs):
        super(DCSNN_Full, self).__init__()
        encoder = lf_generator.base_model
        self.moco = MoCo(encoder, encoder_k, **kwargs)
        self.lf_generator = lf_generator

    def forward(self, im_q, im_k=None):
        kpts_q, localdesc_q, scores_q, scores_map_q = self.lf_generator(im_q) # query: warped
        data = {
            'kpts_q': kpts_q,
            'localdesc_q': localdesc_q,
            'scores_q': scores_q,
            'scores_map_q': scores_map_q
        }
        if im_k is not None:
            kpts_k, localdesc_k, scores_k, scores_map_k = self.lf_generator(im_k) # key: original
            data = { **data, 
                'kpts_k': kpts_k,
                'localdesc_k': localdesc_k,
                'scores_k': scores_k,
                'scores_map_k': scores_map_k,
            }
        
        globaldesc_q, globaldesc_k = self.moco(im_q, im_k)
        data = { **data,
            'globaldesc_q': globaldesc_q,
            'globaldesc_k': globaldesc_k,
        }
        return data

class MoCo(nn.Module):
    """
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, encoder_q, encoder_k, ntrain, dim=128, K=65536, m=0.999, T=0.07, no_moco=False):
        super(MoCo, self).__init__()
        self.encoder_q = encoder_q
        
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T
        self.ntrain = ntrain
        self.activate = not no_moco

        if self.activate:
            print("MoCo activated")
            self.encoder_k = encoder_k
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        
        # # create the queue
        self.register_buffer("queue", torch.randn(dim, K)) # global features
        self.register_buffer("queue_idx", torch.randint(low=0, high=ntrain, size=(1, K))) # ground truth for global feature
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, idxs):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_idx[:, ptr:ptr + batch_size] = idxs
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, im_q, im_k=None):
        q = self.encoder_q(im_q)  # queries: NxC
        if im_k is None:
            return q, None

        if self.activate:
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                k = self.encoder_k(im_k).detach()
             
        else: # not moco
            k = self.encoder_q(im_k).detach()

        return q, k