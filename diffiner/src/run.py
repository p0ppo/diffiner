import os
import argparse
from argparse import ArgumentParser

from .audio.io_handler import AudioRegistry
from .ddrm.backbones.shared import BackboneRegistry
from .ddrm.backbones import dist_util
from .ddrm.diffusion import DiffusionRegistry
from .ddrm.informed_denoiser import get_informed_denoiser


def get_argparse_groups(args, parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


def _run(device, io_handler, model, diffusion):
    model.load_state_dict(
        dist_util.load_state_dict(model.resolve_param_to_load(), map_location="cpu", weights_only=True)
    )
    model.to(device)
    model.eval()

    informed_denoiser = get_informed_denoiser(diffusion)

    noisy = io_handler.get_audio("noisy", "stft", True).to(device)
    proc = io_handler.get_audio("proc", "stft", True).to(device)
    assert noisy.shape == proc.shape

    n_batch, _, _, nf = proc.shape

    noise_stft = noisy - proc
    noise_map = noise_stft.pow(2).sum(1, keepdim=True).pow(1./2.).repeat(1, 2, 1, 1)

    eta_a = 0.9
    eta_b = 0.9

    refined = informed_denoiser(
        model,
        noisy,
        noise_map,
        clip_denoised=False,
        model_kwargs={},
        etaA_ddrm=eta_a,
        etaB_ddrm=eta_b,
    )

    noisy = io_handler.add_dc(noisy)
    proc = io_handler.add_dc(proc)
    refined = io_handler.add_dc(refined)

    io_handler.save(noisy, "noisy.wav", "stft")
    io_handler.save(proc, "proc.wav", "stft")
    io_handler.save(refined, "refined.wav", "stft")


def run():
    parser = ArgumentParser()
    parser.add_argument("--use-gpu", type=bool, default=True)
    parser.add_argument("--audio", type=str, choices=AudioRegistry.get_all_names(), default="base")
    parser.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="unet")
    parser.add_argument("--diffusion", type=str, choices=DiffusionRegistry.get_all_names(), default="spaced")
    tmp_args, _ = parser.parse_known_args()

    if tmp_args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = "cuda"
    else:
        device = "cpu"

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
    arg_groups = get_argparse_groups(args, parser)

    io_handler = audio_cls(
        **vars(arg_groups["Audio"])
    )
    model = backbone_cls(
        sound_class=arg_groups["Audio"].sound_class,
        **vars(arg_groups["Backbone"])
    )
    diffusion = diffusion_cls(
        **vars(arg_groups["Diffusion"])
    )

    _run(device, io_handler, model, diffusion)


if __name__ == "__main__":
    run()
