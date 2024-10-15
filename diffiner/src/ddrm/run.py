import argparse
from argparse import ArgumentParser

from ..
from .backbone.shared import BackboneRegistry
from .diffusion import DiffusionRegistry


def main():
    parser = ArgumentParser()
    parser.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="unet")
    parser.add_argument("--diffusion", type=str, choices=DiffusionRegistry.get_all_names(), default="spaced")
    tmp_args, _ = parser.parse_known_args()

    backbone_cls = BackboneRegistry.get_by_name(tmp_args.backbone)
    diffusion_cls = DiffusionRegistry.get_by_name(tmp_args.diffusion)

    backbone_cls.add_argparse_args(
            parser.add_argument_group("Backbone", description=backbone_cls.__name__))
    diffusion_cls.add_argparse_args(
            parser.add_argument_group("Diffusion", description=diffusion_cls.__name__))

    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser)

    model, diffusion = backbone_cls(), diffusion_cls()

    model.load_state_dict(
            args.model_path, map_location="cpu")
    model.to(device)
    model.eval()
