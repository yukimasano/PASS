import torch
from torchvision.models.resnet import resnet50

import vision_transformer as vits

dependencies = ["torch", "torchvision"]

def dino_vits16(pretrained=True, **kwargs):
    """
    ViT-Small/16x16 pre-trained with DINO for 300 epochs, teacher-temp=0.07, warmup epochs=30, norm-layer=False
    """
    model = vits.__dict__["vit_small"](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/dino_deit_300ep_ttemp0o07_warumup30ep_normlayerF.pth.tar",
            map_location="cpu",
        )['teacher']
        model.load_state_dict(state_dict, strict=False)
    return model


def dino_100ep_vits16(pretrained=True, **kwargs):
    """
    ViT-Small/16x16 pre-trained with DINO.
    """
    model = vits.__dict__["vit_small"](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/dino_deit_100ep.pth.tar",
            map_location="cpu",
        )['teacher']
        model.load_state_dict(state_dict, strict=False)
    return model


def moco_resnet50(pretrained=True, **kwargs):
    """
    ResNet-50 pre-trained with MoCo-v2 for 200epochs
    """
    model = resnet50(pretrained=False, **kwargs)
    model.fc = torch.nn.Identity()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/moco_v2_200ep.pth.tar",
            map_location="cpu",
        )['state_dict']
        model.load_state_dict(state_dict, strict=False)
    return model

def moco_cld_resnet50(pretrained=True, **kwargs):
    """
    ResNet-50 pre-trained with MoCo-v2 for 200epochs
    """
    model = resnet50(pretrained=False, **kwargs)
    model.fc = torch.nn.Identity()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/moco_v2_CLD_200ep.pth.tar",
            map_location="cpu",
        )['state_dict']
        model.load_state_dict(state_dict, strict=False)
    return model

def swav_resnet50(pretrained=True, **kwargs):
    """
    ResNet-50 pre-trained with SwAV for 200 epochs. 2 large crops 6 small ones.
    """
    model = resnet50(pretrained=False, **kwargs)
    model.fc = torch.nn.Identity()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/swav_200ep.pth.tar",
            map_location="cpu",
        )['state_dict']
        model.load_state_dict(state_dict, strict=False)
    return model

