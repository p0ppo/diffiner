import argparse
from argparse import ArgumentParser

from ..audio.io_handler import AudioRegistry
from .backbone.shared import BackboneRegistry
from .backbone import dist_util
from .diffusion import DiffusionRegistry


def main():
    parser = ArgumentParser()
    parser.add_argument("--audio", type=str, choices=AudioRegistry.get_all_names(), default="base")
    parser.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="unet")
    parser.add_argument("--diffusion", type=str, choices=DiffusionRegistry.get_all_names(), default="spaced")
    tmp_args, _ = parser.parse_known_args()

    audio_cls = AudioRegistry.get_by_name(tmp_args.audio)
    backbone_cls = BackboneRegistry.get_by_name(tmp_args.backbone)
    diffusion_cls = DiffusionRegistry.get_by_name(tmp_args.diffusion)

    audio_cls.add_argparse_args(
        parser.add_argument_group("Audio", description=audio_cls.__name__)
    )
    backbone_cls.add_argparse_args(
        parser.add_argument_group("Backbone", description=backbone_cls.__name__)
    )
    diffusion_cls.add_argparse_args(
        parser.add_argument_group("Diffusion", description=diffusion_cls.__name__)
    )

    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser)

    io_handler = audio_cls(
        **vars(arg_groups["Audio"])
    )
    model = backbone_cls(
        sound_class=args.groups["Audio"].sound_class,
        **vars(arg_groups["Backbone"])
    )
    diffusion = diffusion_cls(
        **vars(arg_groups["Diffusion"])
    )

    model.load_state_dict(
        dist_util.load_state_dict(model.resolve_param_to_load(), map_location="cpu")
    )
    model.to(device)
    model.eval()

