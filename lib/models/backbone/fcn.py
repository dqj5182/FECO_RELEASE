import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.fcn import FCNHead


# -------------------------
# Common spec and helpers
# -------------------------
from torchvision.models.resnet import BasicBlock, Bottleneck

_RESNET_FACTORIES = {
    18:  resnet18,
    34:  resnet34,
    50:  resnet50,
    101: resnet101,
    152: resnet152,
}

# your requested spec (note: first channel 64 is the stem, stages are channels[1:])
resnet_spec = {
    18:  (BasicBlock,  [2, 2, 2, 2],  [64,  64, 128, 256,  512], 'resnet18'),
    34:  (BasicBlock,  [3, 4, 6, 3],  [64,  64, 128, 256,  512], 'resnet34'),
    50:  (Bottleneck,  [3, 4, 6, 3],  [64, 256, 512, 1024, 2048], 'resnet50'),
    101: (Bottleneck,  [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
    152: (Bottleneck,  [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152'),
}

def _rswd_for_os(output_stride: int):
    if output_stride == 32: return (False, False, False)
    if output_stride == 16: return (False, True,  False)
    if output_stride == 8:  return (False, True,  True)
    raise ValueError("output_stride must be 8, 16, or 32")

def _effective_rswd(resnet_type: int, output_stride: int):
    # BasicBlock nets do not support dilation
    if resnet_type in (18, 34):
        if output_stride != 32:
            print(f"[FCNResNetEncoder] resnet-{resnet_type} does not support dilation. Forcing output_stride=32.")
        return (False, False, False)
    return _rswd_for_os(output_stride)


# -------------------------
# Encoder: FCN-style ResNet (18,34,50,101,152)
# returns dict with intermediate_feats, final_feat, style_stage_feat
# -------------------------
class FCNResNetEncoder(nn.Module):
    def __init__(self, resnet_type=50, pretrained=True, style_stage=3, output_stride=8):
        super().__init__()
        if resnet_type not in resnet_spec:
            raise ValueError(f"unknown resnet_type {resnet_type}")
        block, layers, channels_all, name = resnet_spec[resnet_type]
        self.name = name
        self.resnet_type = resnet_type
        self.channels_all = channels_all              # includes stem 64
        self.channels = tuple(channels_all[1:])       # stage1..4 only, e.g. (256,512,1024,2048) for 50
        self.style_stage = int(style_stage)

        rswd = _effective_rswd(resnet_type, output_stride)
        backbone = _RESNET_FACTORIES[resnet_type](pretrained=pretrained, replace_stride_with_dilation=rswd)
        backbone.fc = nn.Identity()

        self.body = IntermediateLayerGetter(
            backbone,
            return_layers={'layer1': 'l1', 'layer2': 'l2', 'layer3': 'l3', 'layer4': 'l4'}
        )

    def forward(self, x, return_intermediate=True):
        feats = self.body(x)
        x1, x2, x3, x4 = feats['l1'], feats['l2'], feats['l3'], feats['l4']
        out = {'intermediate_feats': [x1, x2, x3, x4], 'final_feat': x4}
        if return_intermediate:
            out['style_stage_feat'] = {1: x1, 2: x2, 3: x3, 4: x4}[self.style_stage]
        return out

    def init_weights(self):
        import torchvision.models as models
        if self.name == 'resnet18':
            org_resnet = models.resnet18(pretrained=True)
        elif self.name == 'resnet34':
            org_resnet = models.resnet34(pretrained=True)
        elif self.name == 'resnet50':
            org_resnet = models.resnet50(pretrained=True)
        elif self.name == 'resnet101':
            org_resnet = models.resnet101(pretrained=True)
        elif self.name == 'resnet152':
            org_resnet = models.resnet152(pretrained=True)
        else:
            raise ValueError(f"Unsupported model name: {self.name}")

        org_resnet.fc = nn.Identity()
        sd = org_resnet.state_dict()
        sd.pop('fc.weight', None)
        sd.pop('fc.bias', None)
        self.load_state_dict(sd, strict=False)
        print(f"Initialized {self.name} from torchvision pretrained weights")


# -------------------------
# Decoder: main FCN head only
# picks in_channels from resnet_spec so 18/34 use 512 and 50/101/152 use 2048
# accepts dict, list/tuple, or tensor
# -------------------------
class FCNResNetDecoder(nn.Module):
    def __init__(self, resnet_type=50, num_classes=1):
        super().__init__()
        if resnet_type not in resnet_spec:
            raise ValueError(f"unknown resnet_type {resnet_type}")
        # take last stage channel from spec
        in_c = resnet_spec[resnet_type][2][-1]  # channels_all[-1]
        self.main_head = FCNHead(in_c, num_classes)

    def forward(self, enc_out):
        if isinstance(enc_out, dict):
            x = enc_out['final_feat']
        elif isinstance(enc_out, (list, tuple)):
            x = enc_out[-1]
        else:
            x = enc_out
        return self.main_head(x)