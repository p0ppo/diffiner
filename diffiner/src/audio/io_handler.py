import soundfile as sf

from ..util.register import Registry


AudioRegistry = Registry("Audio")


def read_audio(path, target_sr):
    wav, sr = sf.read(path)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav


def write_audio(path, wav, **kwargs):
    sf.write(path, wav, **kwargs)
    return


def wav2stft(wav, fftsize, shiftsize, window):
    stft = torch.stft(
            wav, 
            n_fft=fftsize,
            hop_length=shiftsize,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            pad_mode="reflect")
    return stft


def stft2wav(stft, fftsize, shiftsize, wsize, window):
    # stft: [B, 2, F, T]
    stft_comp = (stft[:, 0, :, :] + 1j * stft[:, 1, :, :]).squeeze()
    wav = torch.istft(
        stft_comp,
        n_fft=fftsize,
        hop_length=shiftsize,
        win_length=wsize,
        window=window,
    )
    return wav


def fold_with_unit_size(tensor, length):
    """
    tensor: [1, 2, F, T]
    output: [num_segments, 2, F, length]
    """
    orig_len = tensor.shape[-1]
    if orig_len > length:
        ratio_nf = int(np.ceil(orig_len / length))
        tensor = tensor.repeat(1, 1, 1, ratio_nf)[..., :length*ratio_nf]
    else:
        ratio_nf = int(np.ceil(length / orig_len))
        tensor = tensor.repeat(1, 1, 1, ratio_nf)[..., :length]
    tensor = tensor.unfold(3, length, length)  # [1, 2, F, length, num_segments]
    tensor[0].permute(3, 0, 1, 2)
    return tensor


def unfold_batch(tensor):
    """
    tensor: [B, 2, F, T]
    output: [1, 2, F, T*B]
    """
    tensor = tensor.permute(1, 2, 3, 0)
    C, F, T, B = tensor.shape
    tensor = tensor.reshape(C, F, T*B).unsqueeze(0)
    return tensor


@AudioRegistry.register("base")
class BaseHandler():
    
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--noisy", type=str, required=True, help="Path to noisy audio file")
        parser.add_argument("--proc", type=str, required=True, help="Path to pre-processed audio file")
        parser.add_argument("--output", type=str, required=True, help="Path to output directory")
        parser.add_argument("--sound-class", type=str, required=True, choices=["dialogue", "music", "effect"], help="Sound class of input audio")

        parser.add_argument("--shiftsize", type=int, default=512, help="Shift size for FFT")
        parser.add_argument("--wsize", type=int, default=1024, help="Window size for FFT")
        parser.add_argument("--fftsize", type=int, default=1024, help="FFT size")
        parser.add_argument("--nf", type=int, default=512, help="Number of frames")
        parser.add_argument("--spec-type", type=str, default="complex", help="Input feature")
        parser.add_argument("--model-sr", type=int, default=48000, help="Sample rate required in model")


    def __init__(
            self, 
            noisy,
            proc,
            output,
            sound_class,
            shiftsize=512,
            wsize=1024,
            fftsize=1024,
            nf=512,
            spec_type="complex",
            model_sr=48000
            ):

        self.noisy = noisy
        self.proc = proc
        self.output = output
        self.sound_class = sound_class

        self.shiftsize = shiftsize
        self.wsize = wsize
        self.fftsize = fftsize
        self.nf = nf
        self.spec_type = spec_type
        self.model_sr = model_sr

        self.noisy_info = sf.info(self.noisy)
        self.proc_info = sf.info(self.proc)
        assert self.noisy_info.samplerate == self.proc_info.samplerate
        assert self.noisy_info.subtype == self.proc_info.subtype


        self.window = nn.Parameter(
                torch.hann_window(self.wsize),
                requires_grad=False,
                )

    def get_audio(self, name, form, fold=True):
        if name == "noisy":
            filename = self.noisy
        elif name == "proc":
            filename = self.proc
        elif name == "reference":
            filename = self.reference
        else:
            raise ValueError(f"Unmatched filename : {name}")

        audio = read_audio(filename, self.model_sr)
        audio = torch.from_numpy(audio)
        if form == "wav":
            return audio
        elif form == "stft":
            stft = wav2stft(audio, self.fftsize, self.shiftsize, self.window)
            stft = stft.permute(0, 3, 1, 2)  # [1, 2, F, T]
            stft = stft[:, :, 1:, :]  # Remove DC
            # TODO: handle stereo input

            self.orig_num_frames = stft.shape[-1]

            if fold:
                stft = fold_with_unit_size(stft, self.nf)
            return stft
    
    def save(self, path, tensor, form="stft"):
        if tensor.device != "cpu":
            tensor = tensor.to("cpu")
        if form == "wav":
            wav = tensor.detach().numpy()
        elif form == "stft":
            stft = unfold_batch(stft)
            wav = stft2wav(stft, self.fftsize, self.shiftsize, self.wsize, self.window)
            wav = wav.detach().numpy()
            write_audio(path, wav, **self.noisy_info)